from functools import partial
from semantic_kernel.functions.kernel_function import KernelFunction, _get_kernel_parameter_metadata_from_config, _invoke_completion, _invoke_stream_completion
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from typing import List, TypeVar, Type, Any, Dict
import yaml

from semantic_kernel.prompt_template.semantic_function_config import SemanticFunctionConfig

T = TypeVar('T')

def kernel_function_yaml_constructor(cls: Type[T]) -> Type[T]:
    cls.from_prompt_yaml = staticmethod(from_prompt_yaml)
    return cls


def from_prompt_yaml(text: str, plugin_name: str, function_name: str) -> KernelFunction:
    """
    Creates a KernelFunction instance from YAML text.

    Args:
        text (str): YAML representation of the SemanticFunctionConfig.
        plugin_name (str): The name of the plugin.
        function_name (str): The name of the function.

    Returns:
        KernelFunction: The kernel function.
    """
    # Convert YAML text to SemanticFunctionConfig object
    function_config_data: Dict[str, Any] = yaml.safe_load(text)
    function_config: SemanticFunctionConfig = SemanticFunctionConfig(**function_config_data)

    if function_config is None:
        raise ValueError("Function configuration cannot be `None`")
    
    semantic_function_params: List[KernelParameterMetadata] = _get_kernel_parameter_metadata_from_config(function_config)

    return KernelFunction(
        function_name=function_name,
        plugin_name=plugin_name,
        description=function_config.description,
        function=partial(_invoke_completion, function_config=function_config),
        parameters=semantic_function_params,
        return_parameter=KernelParameterMetadata(
            name="return",
            description="The completion result",
            default_value=None,
            type="FunctionResult",
            required=True,
        ),
        stream_function=partial(_invoke_stream_completion, function_config=function_config) if function_config.has_streaming else None,
        is_semantic=True,
        chat_prompt_template=function_config.prompt_template if function_config.has_chat_prompt else None,
    )
