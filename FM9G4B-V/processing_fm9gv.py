# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for FM9GV.
"""

from typing import List, Optional, Union, Dict, Any
import torch
import re

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType, requires_backends, is_torch_dtype, is_torch_device

from .image_processing_fm9gv import FM9GVBatchFeature


class FM9GVProcessor(ProcessorMixin):
    r"""
    Constructs a FM9GV processor which wraps a FM9GV image processor and a FM9GV tokenizer into a single processor.

    [`FM9GVProcessor`] offers all the functionalities of [`FM9GVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~FM9GVProcessor.__call__`] and [`~FM9GVProcessor.decode`] for more information.

    Args:
        image_processor ([`FM9GVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.version = image_processor.version
    
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        use_image_id: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs
    ) -> FM9GVBatchFeature:

        if images is not None:
            image_inputs = self.image_processor(images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors)
        return self._convert_images_texts_to_inputs(image_inputs, text, max_slice_nums=max_slice_nums, use_image_id=use_image_id, max_length=max_length, **kwargs)
    
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text
        # return self.tokenizer.batch_decode(*args, **kwargs)
    
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or (hasattr(self.tokenizer, "eot_id") and result[-1] == self.tokenizer.eot_id):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(
        self, input_str, max_inp_length: Optional[int] = None
    ):
        input_ids = self.tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_tokens = torch.where(start_cond)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        image_bounds = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )
        return input_ids, image_bounds

    def _convert_images_texts_to_inputs(
            self, 
            images, 
            texts: Union[str, List[str]], 
            truncation=None, 
            max_length=None,
            max_slice_nums=None,
            use_image_id=None, 
            return_tensors=None,
            **kwargs
        ):
        if images is None or not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs)
            return FM9GVBatchFeature(data={**model_inputs})
        
        pattern = "(<image>./</image>)"
        images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        
        if isinstance(texts, str):
            texts = [texts]
        input_ids_list = []
        image_bounds_list = []
        for index, text in enumerate(texts):
            image_tags = re.findall(pattern, text)
            assert len(image_tags) == len(image_sizes[index])
            text_chunks = text.split(pattern)
            final_text = ""
            for i in range(len(image_tags)):
                final_text = final_text + text_chunks[i] + \
                    self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][i], 
                        i,
                        max_slice_nums,
                        use_image_id
                    )
            final_text += text_chunks[-1]
            input_ids, image_bounds = self._convert(final_text, max_length)
            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
        padded_input_ids, padding_lengths = self.pad(
            input_ids_list,
            padding_side="left"
        )
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            attention_mask[i, :length] = False

        return FM9GVBatchFeature(data={
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes
        })

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length
