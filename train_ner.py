import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import random
import os

# --- Step 1: Import the training data ---
# This line imports the TRAIN_DATA list from your other file.
try:
    from training_data import TRAIN_DATA
except ImportError:
    print("Error: 'training_data.py' not found.")
    print("Please make sure the training data file is in the same directory.")
    exit()

def train_spacy_model(
    data,
    output_dir="./models/custom_redaction_model",
    iterations=30,
    dropout=0.5
):
    """
    Loads the training data, trains a new spaCy NER model, and saves it.

    Args:
        data (list): The TRAIN_DATA from training_data.py.
        output_dir (str): The directory to save the trained model.
        iterations (int): The number of training iterations.
        dropout (float): The dropout rate for regularization.
    """
    
    # --- Step 2: Prepare the data and environment ---
    # Create a new, blank English model.
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    # Create a new NER pipe and add it to the pipeline.
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # --- Step 3: Add all unique entity labels to the NER pipe ---
    # This part automatically finds every unique label (e.g., "NAME", "IP_ADDRESS")
    # from your training data and adds it to the model.
    labels = set()
    for _, annotations in data:
        for ent in annotations.get("entities"):
            labels.add(ent[2])
    
    for label in labels:
        ner.add_label(label)
    
    print(f"Added {len(labels)} unique labels to the NER model: {sorted(list(labels))}")

    # --- Step 4: Begin the training process ---
    # Disable other pipes during training for efficiency.
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.select_pipes(disable=unaffected_pipes):
        optimizer = nlp.begin_training()
        
        print("\n--- Starting Training ---")
        for itn in range(iterations):
            random.shuffle(data)
            losses = {}
            
            # Batch up the examples using spaCy's batcher
            for batch in spacy.util.minibatch(data, size=2):
                examples = []
                for text, annots in batch:
                    # Create an Example object for each training instance
                    examples.append(Example.from_dict(nlp.make_doc(text), annots))
                
                # Update the model with the batch of examples
                nlp.update(
                    examples,
                    drop=dropout,
                    sgd=optimizer,
                    losses=losses,
                )
            
            # Print the loss for each iteration
            print(f"Iteration: {itn + 1}/{iterations}, Losses: {losses}")

    # --- Step 5: Save the trained model to disk ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save the final trained model. This model can be loaded later for redaction.
    # We save it to a subdirectory called 'model-best' as is standard.
    model_path = os.path.join(output_dir, "model-best")
    nlp.to_disk(model_path)
    print(f"\n--- Training Complete ---")
    print(f"✅ Model saved to: {model_path}")


if __name__ == "__main__":
    # When the script is run directly, start the training process.
    train_spacy_model(TRAIN_DATA)
