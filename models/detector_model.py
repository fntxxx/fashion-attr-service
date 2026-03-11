import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

_processor = None
_model = None


def get_detector():
    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()

    return _processor, _model


def run_detection(image, text_prompt: str, box_threshold=0.28, text_threshold=0.22):
    processor, model = get_detector()

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )

    return results[0]