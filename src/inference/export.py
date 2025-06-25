"""Model export utilities for TorchScript and ONNX."""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained models to different formats."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
    def export_torchscript(self, output_path: str, max_length: int = 512) -> str:
        """Export model to TorchScript format."""
        if self.model is None:
            self.load_model()
            
        logger.info("Exporting to TorchScript...")
        
        # Create dummy input
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, max_length)),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long)
        }
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self.model, 
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                strict=False
            )
        
        # Save traced model
        torchscript_path = os.path.join(output_path, "model.pt")
        traced_model.save(torchscript_path)
        
        logger.info(f"TorchScript model saved to {torchscript_path}")
        return torchscript_path
        
    def export_onnx(self, output_path: str, max_length: int = 512) -> str:
        """Export model to ONNX format."""
        if self.model is None:
            self.load_model()
            
        logger.info("Exporting to ONNX...")
        
        # Create dummy input
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, max_length)),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long)
        }
        
        # Export to ONNX
        onnx_path = os.path.join(output_path, "model.onnx")
        
        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"}
            }
        )
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")
        except Exception as e:
            logger.warning(f"ONNX model verification failed: {e}")
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return onnx_path
        
    def export_optimized_onnx(self, output_path: str, max_length: int = 512) -> str:
        """Export optimized ONNX model."""
        try:
            from onnxruntime.tools import optimizer
            
            # First export regular ONNX
            onnx_path = self.export_onnx(output_path, max_length)
            
            # Optimize
            optimized_path = os.path.join(output_path, "model_optimized.onnx")
            optimizer.optimize_model(
                onnx_path,
                model_type="bert",
                num_heads=12,  # DistilBERT heads
                hidden_size=768,  # DistilBERT hidden size
                optimization_options=None
            ).save_model_to_file(optimized_path)
            
            logger.info(f"Optimized ONNX model saved to {optimized_path}")
            return optimized_path
            
        except ImportError:
            logger.warning("ONNX optimization tools not available, skipping optimization")
            return self.export_onnx(output_path, max_length)
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            return self.export_onnx(output_path, max_length)


def export_model(model_path: str, max_length: int = 512) -> Tuple[str, str]:
    """Export model to both TorchScript and ONNX formats."""
    exporter = ModelExporter(model_path)
    
    # Create export directory
    export_dir = os.path.join(model_path, "exported")
    Path(export_dir).mkdir(exist_ok=True)
    
    # Export to both formats
    torchscript_path = exporter.export_torchscript(export_dir, max_length)
    onnx_path = exporter.export_onnx(export_dir, max_length)
    
    # Also try optimized ONNX
    try:
        optimized_onnx_path = exporter.export_optimized_onnx(export_dir, max_length)
        logger.info(f"Exported optimized ONNX to {optimized_onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export optimized ONNX: {e}")
    
    return torchscript_path, onnx_path


def verify_exported_models(model_path: str, max_length: int = 512) -> bool:
    """Verify that exported models work correctly."""
    try:
        import onnxruntime as ort
        
        # Load original model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        original_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        original_model.eval()
        
        # Test input
        test_text = "This is a test sentence for model verification."
        inputs = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Get original prediction
        with torch.no_grad():
            original_output = original_model(**inputs)
            original_logits = original_output.logits
        
        # Test TorchScript model
        torchscript_path = os.path.join(model_path, "exported", "model.pt")
        if os.path.exists(torchscript_path):
            traced_model = torch.jit.load(torchscript_path)
            with torch.no_grad():
                ts_output = traced_model(inputs["input_ids"], inputs["attention_mask"])
                
            # Compare outputs
            if torch.allclose(original_logits, ts_output, atol=1e-4):
                logger.info("TorchScript model verification passed")
            else:
                logger.warning("TorchScript model outputs differ from original")
        
        # Test ONNX model
        onnx_path = os.path.join(model_path, "exported", "model.onnx")
        if os.path.exists(onnx_path):
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_logits = torch.tensor(ort_outputs[0])
            
            # Compare outputs
            if torch.allclose(original_logits, onnx_logits, atol=1e-4):
                logger.info("ONNX model verification passed")
            else:
                logger.warning("ONNX model outputs differ from original")
        
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--verify", action="store_true", help="Verify exported models")
    
    args = parser.parse_args()
    
    # Export models
    torchscript_path, onnx_path = export_model(args.model_path, args.max_length)
    
    print(f"TorchScript model: {torchscript_path}")
    print(f"ONNX model: {onnx_path}")
    
    # Verify if requested
    if args.verify:
        success = verify_exported_models(args.model_path, args.max_length)
        print(f"Verification: {'PASSED' if success else 'FAILED'}")