import warnings
from typing import Any, Dict, List, Union
import numpy as np

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args

if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    )

logger = logging.get_logger(__name__)

Prediction = Dict[str, Any]
Predictions = List[Prediction]

@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageSegmentationPipeline(Pipeline):
    """
    Optimized Image segmentation pipeline using any `AutoModelForXXXSegmentation`.
    This pipeline predicts masks of objects and their classes, with support for batch processing and per-image thresholds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)
        self.check_model_type(mapping)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}

        if "subtask" in kwargs:
            preprocess_kwargs["subtask"] = kwargs["subtask"]
            postprocess_kwargs["subtask"] = kwargs["subtask"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "overlap_mask_area_threshold" in kwargs:
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]
        if "timeout" in kwargs:
            warnings.warn(
                "The `timeout` argument is deprecated and will be removed in version 5 of Transformers", FutureWarning
            )
            preprocess_kwargs["timeout"] = kwargs["timeout"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def __call__(self, inputs=None, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.
        Supports batch processing, with per-image thresholds if needed.
        """
        if inputs is None:
            raise ValueError("Inputs cannot be None for image segmentation!")

        if isinstance(inputs, list) and len(inputs) > 1:
            return self._batch_process(inputs, **kwargs)

        return super().__call__(inputs, **kwargs)

    def _batch_process(self, inputs: List[Union[str, Image.Image]], **kwargs) -> List[Predictions]:
        """
        Process multiple images at once for batch segmentation.
        """
        results = []
        for image in inputs:
            results.append(self.__call__(image, **kwargs))
        return results

    def preprocess(self, image: Union[str, Image.Image], subtask=None, timeout=None):
        image = load_image(image, timeout=timeout)
        target_size = [(image.height, image.width)]
        
        if self.model.config.__class__.__name__ == "OneFormerConfig":
            kwargs = {"task_inputs": [subtask]} if subtask else {}
            inputs = self.image_processor(images=[image], return_tensors="pt", **kwargs)
            if self.framework == "pt":
                inputs = inputs.to(self.torch_dtype)
            inputs["task_inputs"] = self.tokenizer(
                inputs["task_inputs"],
                padding="max_length",
                max_length=self.model.config.task_seq_len,
                return_tensors=self.framework,
            )["input_ids"]
        else:
            inputs = self.image_processor(images=[image], return_tensors="pt")
            if self.framework == "pt":
                inputs = inputs.to(self.torch_dtype)
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs: Dict[str, Any]):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size
        return model_outputs

    def postprocess(
        self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
    ) -> List[Prediction]:
        fn = None

        if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            fn = self.image_processor.post_process_panoptic_segmentation
        elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
            fn = self.image_processor.post_process_instance_segmentation
        elif subtask in {"semantic", None} and hasattr(self.image_processor, "post_process_semantic_segmentation"):
            fn = self.image_processor.post_process_semantic_segmentation
        else:
            raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")

        outputs = fn(
            model_outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            target_sizes=model_outputs["target_size"],
        )[0]

        return self._process_annotations(outputs)

    def _process_annotations(self, outputs: Dict[str, Any]) -> List[Prediction]:
        """
        Helper function to convert model output to list of annotations.
        """
        annotation = []
        segmentation = outputs["segmentation"]

        for segment in outputs.get("segments_info", []):
            mask = (segmentation == segment["id"]) * 255
            mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
            label = self.model.config.id2label.get(segment["label_id"], "Unknown")
            score = segment.get("score", None)
            annotation.append({"score": score, "label": label, "mask": mask})

        return annotation
