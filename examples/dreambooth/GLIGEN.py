import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config, instantiate_from_config_text_encoder
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path, device):
    
    saved_ckpt = torch.load(ckpt_path,map_location=device)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config_text_encoder(config['text_encoder'], device).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(device, model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device) # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).to(device)  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').to(device).T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['pixel_values'] = torch.ones(1,3,224,224).to(device) # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(device,meta, model, processor, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    # version = "/hhd2/home/Code/lzl/GLIGEN/clip-vit-large-patch14"
    # model = CLIPModel.from_pretrained(version).cuda()
    # processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(device, model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(device, model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb



class Gligen_Video():
    def __init__(self, args, device, ckpt, version = "/hhd2/home/Code/lzl/GLIGEN/clip-vit-large-patch14"):
        self.args = args
        self.device = device
        self.model, self.autoencoder, self.text_encoder, self.diffusion, self.config = load_ckpt(ckpt, device)
        self.clip_model = CLIPModel.from_pretrained(version).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(version)
        
    def init_video(self, text_prompt, output_folder,ref_image = None, phrases = None):
        """ init the basic setting of a video sequence

        Args:
            ref_image (list): 
            text_prompt (str): 
            output_folder (folder_path): 
            phrases: text reference inpaint
        """
        self.prompt = text_prompt 
        self.ref_image = ref_image
        self.output_folder = output_folder
        self.phrases = phrases
    def mask2box(self, mask_image):
        # 转换图像为numpy数组
        img = Image.open(mask_image)
        mask_array = np.array(img)

        # 将所有mask转换为二值的
        binary_masks = [(mask_array == i) * 1 for i in range(1, np.max(mask_array) + 1)]
        boxes = []
        for binary_mask in binary_masks:
            # 找到所有非零像素的x和y坐标
            y_indices, x_indices = np.nonzero(binary_mask)

            # 计算矩形框的坐标
            x_min = np.min(x_indices)
            x_max = np.max(x_indices)
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            # 存储矩形框的坐标
            boxes.append([x_min / img.size[0], y_min / img.size[1], x_max / img.size[0], y_max / img.size[1]])

        return boxes
        
    @torch.no_grad()
    def run_frame(self, mask_image, back_image, starting_noise=None):
        """
        This function inpaint a frame based on a reference image.

        args:
            mask_image (path): mask from TAM
            back_image (path): original frame
        
        Returns:
            
    """
        device = self.device
        # - - - - - prepare models - - - - - # 
        model, autoencoder, text_encoder, diffusion, config = self.model, self.autoencoder, self.text_encoder, self.diffusion, self.config

        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
        model.grounding_tokenizer_input = grounding_tokenizer_input
        
        grounding_downsampler_input = None
        if "grounding_downsampler_input" in config:
            grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])
        # - - - - - update config from args - - - - - # 
        config.update( vars(self.args) )
        config = OmegaConf.create(config)

        # - - - - - prepare batch - - - - - #
        # convert mask to bounding box, can only deal with one mask.
        boxes = self.mask2box(mask_image)
        meta = {"images":self.ref_image, "locations": boxes, "phrases":self.phrases}
        if self.ref_image != None:
            assert len(boxes) == len(self.ref_image), "The number of mask != the number of reference images."
        if self.phrases != None:
            assert len(boxes) == len(self.phrases), "The number of mask != the number of reference text."
        
        batch = prepare_batch(device, meta,self.clip_model, self.clip_processor, config.batch_size)
        context = text_encoder.encode(  [self.prompt]*config.batch_size  )
        uc = text_encoder.encode( config.batch_size*[""] )
        if self.args.negative_prompt is not None:
            uc = text_encoder.encode( config.batch_size*[self.args.negative_prompt] )

        # - - - - - sampler - - - - - # 
        alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type")) ########
        if config.no_plms:
            sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            steps = 250 
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            steps = 50 

        # - - - - - inpainting related - - - - - #
        inpainting_mask = z0 = None  # used for replacing known region in diffusion process
        inpainting_extra_input = None # used as model input 
        
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).to(device)
        img = Image.open(back_image)
        img = np.array(img)
        h, w = img.shape[:2]
        input_image = F.pil_to_tensor( Image.open(back_image).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).to(device) / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        

        # - - - - - input for gligen - - - - - #
        grounding_input = grounding_tokenizer_input.prepare(batch)
        grounding_extra_input = None
        if grounding_downsampler_input != None:
            grounding_extra_input = grounding_downsampler_input.prepare(batch)

        input = dict(
                    x = starting_noise, 
                    timesteps = None, 
                    context = context, 
                    grounding_input = grounding_input,
                    inpainting_extra_input = inpainting_extra_input,
                    grounding_extra_input = grounding_extra_input,

                )


        # - - - - - start sampling - - - - - #
        shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
        samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
        samples_fake = autoencoder.decode(samples_fake)

        # - - - - - save - - - - - #
        output_folder = self.output_folder
        os.makedirs( output_folder, exist_ok=True)

        start = len( os.listdir(output_folder) )
        image_ids = list(range(start,start+config.batch_size))
        print(image_ids)
        for image_id, sample in zip(image_ids, samples_fake):
            img_name = '{:08d}.png'.format(image_id)
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample = sample.resize((w, h))
            sample.save(  os.path.join(output_folder, img_name)   )
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    args = parser.parse_args()
    device = "cuda:4"
    #load ckpt, 需要传一个args对象包含以上入参
    checkpoint_path = "/hhd2/home/Code/lzl/GLIGEN/gligen_checkpoints/checkpoint_inpainting_text_image.pth"
    Gligen_model = Gligen_Video(args, device , checkpoint_path)
    
    # init the basic info of video sequence: reference images(list), text prompt, output folder
    
    # 可以根据mask中顺序传入多个reference images和phrases,也可只传一种
    ref_images = ['inference_images/bottle.jpg', 'inference_images/bigben.jpg']
    phrases = ['bottle','bigben']
    
    Gligen_model.init_video("a tree beside the mirror and a rubbish bin on the table","output", ref_image=ref_images, phrases=phrases)
    
    Gligen_model.run_frame("inference_images/red_book_mask.png","inference_images/red_book_result.png")


    



