"""
AutoLyrics — PEFT LoRA configuration for Whisper.
Apply LoRA to encoder and decoder attention projection layers.
"""
from peft import LoraConfig, TaskType, get_peft_model
from model import PitchAwareWhisper


LORA_CONFIG = LoraConfig(
    task_type   = TaskType.SEQ_2_SEQ_LM,
    r           = 8,           # rank — increase to 16 for larger datasets
    lora_alpha  = 16,          # scaling: alpha / r = 2
    lora_dropout= 0.1,
    target_modules = [           # Whisper attention projections
        "q_proj", "v_proj",    # encoder + decoder self-attn
        "out_proj",             # output projections
        "encoder_attn.q_proj",
        "encoder_attn.v_proj",
    ],
    bias         = "none",
    modules_to_save = ["pitch_proj", "alpha"],  # always train these fully
)


def build_lora_model(model_name: str = "openai/whisper-small"):
    base  = PitchAwareWhisper(model_name)
    model = get_peft_model(base, LORA_CONFIG)
    model.print_trainable_parameters()
    return model