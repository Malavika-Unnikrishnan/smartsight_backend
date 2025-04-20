import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define function for image captioning
def analyze_scene_api(image):
    try:
        # Preprocess the image
        processed_image = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output = model.generate(processed_image)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

# Create Gradio Interface with API mode
interface = gr.Interface(
    fn=analyze_scene_api,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="BLIP API for Image Captioning",
    description="Send an image to get a caption response in JSON format."
)

if __name__ == "__main__":
    # Launch Gradio interface in API mode
    interface.launch(server_name="0.0.0.0", server_port=7860)
