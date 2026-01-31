import os
import sys
from src.logging import logging
from src.exception import CustomException
from src.chemical_analysis.component.pre_training04 import PreTraining
from src.chemical_analysis.component.fine_tuning05 import FineTuning


class TrainingController:
    def __init__(self):
        self.pretrained_models_dir = "artifacts/models"
        self.finetuned_models_dir = "artifacts/finetuned_models"
        self.target_cols = ["N", "P", "K", "pH"]

    def check_pretrained_model_exists(self) ->bool:
        if not os.path.exists(self.pretrained_models_dir):
            return False
        
        for target in self.target_cols:
            model_path = os.path.join(self.pretrained_models_dir, f"xgb_{target}.joblib")
            if not os.path.exists(model_path):
                return False
        return True
    
    def check_finetuned_model_exists(self) ->bool:
        if not os.path.exists(self.finetuned_models_dir):
            return False
        
        for target in self.target_cols:
            model_path = os.path.join(self.finetuned_models_dir, f"xgb_{target}.joblib")
            if not os.path.exists(model_path):
                return False
        return True   
    
    def run_pretraining(self, force: bool = False):
        try:
            if not force and self.check_pretrained_model_exists():
                logging.info("Pretrined weights present, use force = True to train")
                return False
            pretrainer = PreTraining(
                X_path="artifacts/X_lucas.npy",
                y_path="artifacts/y_lucas.npy",
                params_path="params.yaml",
                models_dir=self.pretrained_models_dir,
                metrics_path="artifacts/metrics.txt"
            )
            pretrainer.train_and_save()
            logging.info("Pretrainig Completed successflly")
            return True
        except Exception as e:
            logging.error("Error in model pretraining")
            raise CustomException(e, sys)
    
    def run_finetuning(self, force: bool = False):
        try:
            if not self.check_pretrained_model_exists():
                raise FileNotFoundError("Pretrained models not found")
            
            if not force and self.check_finetuned_model_exists():
                logging.info("Fine Tune model exists, use force = True to run")
                return False
            finetuner = FineTuning(
                X_path="artifacts/X_punjab.npy",
                y_path="artifacts/y_punjab.npy",
                params_path="params.yaml",
                pretrained_models_dir=self.pretrained_models_dir,
                finetuned_models_dir=self.finetuned_models_dir,
                metrics_path="artifacts/finetune_metrics.txt"
            )
            finetuner.finetune_and_save()
            return True
        except Exception as e:
            logging.error("Error during fine tuning")
            raise CustomException(e, sys)
    
    def run_training_pipeline(self, force_pretrined: bool= False, force_finetuned: bool = False):
        try:
            pretrain = self.run_pretraining(force= force_pretrined)
            finetune = self.run_finetuning(force= force_finetuned)

            return {
                "pretrain_executed": pretrain,
                "finetune_executed": finetune
            }
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    controller = TrainingController()
    controller.run_training_pipeline(False, True)