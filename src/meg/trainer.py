import torch
from trl import SFTTrainer


class MEGTrainer(SFTTrainer):
    def __init__(self, task, **kwargs):
        super(MEGTrainer, self).__init__(**kwargs)
        self.task = task
        self.counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        def _adjust_label_size(preds, labels):
            size_increase = preds.shape[1] - labels.shape[1]
            labels = torch.cat((
                -100*torch.ones((labels.shape[0], size_increase), dtype=torch.int, device=labels.device), labels), 
            dim=1)
            return labels 
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            labels = _adjust_label_size(inputs["input_ids"], labels)
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    