import dataclasses
import logging
from typing import TYPE_CHECKING

from typing_extensions import override
import safetensors.torch

from openpi.models import model as _model
from openpi.models import pi0_config

if TYPE_CHECKING:
    from openpi.models.pi0grounded import Pi0Grounded

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0GroundedConfig(pi0_config.Pi0Config):
    """Configuration for Pi0Grounded/Pi05Grounded model with multi-chunk action generation.
    
    This extends Pi0Config to add the ability to generate multiple action chunks
    in a single inference call, reducing the need for frequent model queries.
    
    Works with both Pi0 and Pi05 architectures - set pi05=True for Pi05Grounded behavior.
    
    Key features:
    - num_action_chunks: Number of action chunks to generate per inference (default: 1)
    - When num_action_chunks > 1, the model generates multiple independent action sequences
      that can be executed sequentially with less frequent replanning.
    - Supports both Pi0 (pi05=False) and Pi05 (pi05=True) architectures
    - Output shape: [batch_size, num_action_chunks, action_horizon, action_dim]
    
    The default num_action_chunks=1 preserves the original Pi0/Pi05 behavior exactly.
    
    Examples:
        # Pi0Grounded with 3 chunks (outputs [batch, 3, 50, action_dim])
        Pi0GroundedConfig(pi05=False, num_action_chunks=3, action_horizon=50, ...)
        
        # Pi05Grounded with 3 chunks (outputs [batch, 3, 50, action_dim])
        Pi0GroundedConfig(pi05=True, num_action_chunks=3, action_horizon=50, ...)
    """
    
    # Number of action chunks to generate during inference
    # Default 1 preserves original Pi0/Pi05 behavior
    num_action_chunks: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if self.num_action_chunks < 1:
            raise ValueError(f"num_action_chunks must be >= 1, got {self.num_action_chunks}")
    
    @property
    @override
    def model_type(self) -> _model.ModelType:
        """The model type."""
        return _model.ModelType.PI0_GROUNDED
    
    @override
    def create(self, rng):
        from openpi.models.pi0grounded import Pi0Grounded
        import flax.nnx as nnx
        
        return Pi0Grounded(self, rngs=nnx.Rngs(rng))
    
    @override
    def load_pytorch(self, train_config, weight_path: str):
        """Load a PyTorch Pi0Grounded/Pi05Grounded model from checkpoint.
        
        This loads the base Pi0/Pi05 weights and creates a Pi0Grounded model that
        can generate multiple action chunks during inference.
        """
        from openpi.models_pytorch.pi0grounded_pytorch import PI0GroundedPytorch
        
        model_variant = "Pi05Grounded" if self.pi05 else "Pi0Grounded"
        logger.info(f"Loading {model_variant} PyTorch model with num_action_chunks={self.num_action_chunks}")
        logger.info(f"train_config: {train_config}")
        model = PI0GroundedPytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model
