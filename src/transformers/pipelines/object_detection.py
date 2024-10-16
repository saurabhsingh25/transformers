from collections import Counter
import warnings
from typing import Any, Dict, List, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args

if is_vision_available():
    from ..image_utils import load_image

if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import (
        MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    )

logger = logging.get_logger(__name__)

Prediction = Dict[str, Any]
Predictions = List[Prediction]

@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.
    
    Example:
    ```python
    >>> from transformers import pipeline
    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, 
    {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]
    ```

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"{self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
        self.check_model_type(mapping)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        return preprocess_params, {}, postprocess_kwargs

    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image`, `List[PIL.Image]`):
                The pipeline handles three types of images:
                - A string containing an HTTP(S) link pointing to an image.
                - A string containing a local path to an image.
                - A PIL image or list of PIL images.

            threshold (`float`, optional, defaults to 0.5):
                The probability threshold for predictions.
            top_k (`int`, optional, defaults to None):
                Limit to the top K predictions per image.

        Returns:
            A list of dictionaries with prediction results.
        """
        if "images" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("images")
        return super().__call__(*args, **kwargs)

    def preprocess(self, image, timeout=None):
        image = load_image(image)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.image_processor(images=[image], return_tensors="pt")

        if self.framework == "pt":
            inputs = inputs.to(self.torch_dtype)
        if self.tokenizer is not None:
            inputs = self.tokenizer(text=inputs["words"], boxes=inputs["boxes"], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        outputs = self.model(**model_inputs)
        model_outputs = outputs.__class__({"target_size": target_size, **outputs})
        if self.tokenizer is not None:
            model_outputs["bbox"] = model_inputs["bbox"]
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.5, top_k=None):
        target_size = model_outputs["target_size"]
        height, width = target_size[0].tolist()

        if self.tokenizer is not None:
            # For LayoutLM models with token classification.
            def unnormalize(bbox):
                return self._get_bounding_box(torch.Tensor([
                    (width * bbox[0] / 1000),
                    (height * bbox[1] / 1000),
                    (width * bbox[2] / 1000),
                    (height * bbox[3] / 1000),
                ]))

            scores, classes = model_outputs["logits"].squeeze(0).softmax(dim=-1).max(dim=-1)
            labels = [self.model.config.id2label[prediction] for prediction in classes.tolist()]
            boxes = [unnormalize(bbox) for bbox in model_outputs["bbox"].squeeze(0)]
            annotations = [dict(score=score, label=label, box=box) 
                           for score, label, box in zip(scores.tolist(), labels, boxes) if score > threshold]

        else:
            # Regular Object Detection Model.
            raw_annotations = self.image_processor.post_process_object_detection(model_outputs, threshold, target_size)
            raw_annotation = raw_annotations[0]
            scores, labels, boxes = raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"]

            raw_annotation["scores"] = scores.tolist()
            raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
            raw_annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

            annotations = [dict(score=score, label=label, box=box)
                           for score, label, box in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"])]

        # Limit results to top_k if specified.
        if top_k:
            annotations = sorted(annotations, key=lambda x: -x["score"])[:top_k]

        return annotations

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Convert [xmin, xmax, ymin, ymax] to {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax} format.

        Args:
            box (`torch.Tensor`): Tensor with coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Bounding box in dictionary format.
        """
        if self.framework != "pt":
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
