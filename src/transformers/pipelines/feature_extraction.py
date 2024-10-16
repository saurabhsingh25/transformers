from collections import Counter
from typing import Dict, List, Union

from ..utils import add_end_docstrings
from .base import GenericTensor, Pipeline, build_pipeline_init_args


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
            Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.
        return_token_count (`bool`, *optional*):
            If `True`, returns a count of the tokens in the input in addition to the features.
    """,
)
class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline
    >>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension]
    torch.Size([1, 8, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.
    """

    def _sanitize_parameters(
        self, truncation=None, tokenize_kwargs=None, return_tensors=None, return_token_count=None, **kwargs
    ):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs
        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors
        if return_token_count is not None:
            postprocess_params["return_token_count"] = return_token_count

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: Union[str, List[str]], **tokenize_kwargs) -> Dict[str, GenericTensor]:
        if not hasattr(self, 'tokenizer'):
            raise ValueError("Tokenizer is missing. Please load a valid tokenizer.")

        model_inputs = self.tokenizer(inputs, return_tensors=self.framework, **tokenize_kwargs)
        
        # Add token count information
        token_count = Counter(model_inputs['input_ids'][0].tolist())  # Count tokens in the first sequence
        model_inputs['token_count'] = token_count

        return model_inputs

    def _forward(self, model_inputs):
        if self.framework not in ["pt", "tf"]:
            raise ValueError(f"Framework '{self.framework}' is not supported. Use 'pt' (PyTorch) or 'tf' (TensorFlow).")
        
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, return_tensors=False, return_token_count=False):
        # [0] is the first available tensor, logits or last_hidden_state.
        features = model_outputs[0]
        result = features if return_tensors else features.tolist()

        # Return token count if requested
        if return_token_count and "token_count" in model_outputs:
            return {"features": result, "token_count": model_outputs["token_count"]}
        
        return result

    def __call__(self, *args, return_token_count=False, **kwargs):
        """
        Extract the features of the input(s) and optionally return the token count.

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.
            return_token_count (`bool`, *optional*): Whether to return the count of tokens in the input.

        Return:
            A nested list of `float`: The features computed by the model, and optionally the token count.
        """
        return super().__call__(*args, return_token_count=return_token_count, **kwargs)
