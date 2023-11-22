import torch
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments
import argparse

class SentimentModel(torch.nn.Module):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        self.sentiment_regressor = torch.nn.Linear(self.config.hidden_size, 2)
        self.loss_fc = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, input_ids, attention_mask, sentiment, *args, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        last_hidden = out['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor(last_hidden[:,0,:].view(-1, self.model.config.hidden_size))
        
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        sentiment_loss = self.loss_fc(softmaxed, sentiment)
        
        return {"model_output": softmaxed,
                "loss": sentiment_loss + out.loss}
        
    def predict(self, input_ids=None, *args, **kwargs):
        
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        
        return softmaxed
    
    def validate(self, input_ids=None, sentiment=None,*args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        
        softmaxed = softmaxed.flatten().cpu()
        
        predicted_val = torch.argmax(softmaxed)
        
        label_val = torch.argmax(sentiment.flatten())
        
        validation_true_val = (label_val == predicted_val).float().sum().numpy()        
        
        return {"acc" : validation_true_val}
    
    
parser = argparse.ArgumentParser()

parser.add_argument("--model", default='roberta-base', type=str, help="Default model")

parser.add_argument("--epochs", default=4, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")

def main(args):
    model = SentimentModel(args.model)
    
    training_args = TrainingArguments(
        save_strategy='no',
        warmup_steps= 4 * len(['future_data']) / args.batch_size,
        logging_steps=100,
        weight_decay = 0.0,
        num_train_epochs = args.epochs,
        learning_rate = 5e-5,
        fp16 = True if torch.cuda.is_available() else False,
        ddp_backend = "nccl",
        lr_scheduler_type="cosine",
        logging_dir = './logs',
        output_dir = './results',
        per_device_train_batch_size = args.batch_size    
    )
    
    Trainer(model=model, args=training_args,train_dataset=None, data_collator=None).train()
    


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
