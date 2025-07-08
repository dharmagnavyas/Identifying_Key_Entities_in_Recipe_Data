# Recipe Named Entity Recognition (NER) Model

A machine learning solution for automatically extracting ingredients, quantities, and units from recipe text using Conditional Random Fields (CRF).

## Overview

This project implements a Named Entity Recognition system specifically designed for recipe data processing. The model automatically identifies and classifies three key components in recipe text:

- **Ingredients** (flour, salt, tomatoes)
- **Quantities** (2, 1/2, three)  
- **Units** (cups, tablespoons, grams)

## Performance

- **Overall Accuracy**: 99.83%
- **Ingredient Recognition**: 100% accuracy
- **Quantity Detection**: 99.51% accuracy
- **Unit Classification**: 99.16% accuracy

## Quick Start

### Prerequisites

```bash
pip install sklearn-crfsuite pandas numpy matplotlib seaborn scikit-learn
```

Optional (for enhanced features):
```bash
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
import joblib
from recipe_ner import predict_recipe_entities, extract_recipe_components

# Load trained model
model = joblib.load('crf_model.pkl')

# Predict entities in recipe text
recipe_text = "2 cups flour 1 teaspoon salt 3 tablespoons olive oil"
entities = predict_recipe_entities(recipe_text, model)

# Extract structured components
components = extract_recipe_components(recipe_text, model)
print(f"Ingredients: {components['ingredients']}")
print(f"Quantities: {components['quantities']}")  
print(f"Units: {components['units']}")
```

## Project Structure

```
recipe-ner/
├── data/
│   └── ingredient_and_quantity.json     # Training dataset
├── models/
│   └── crf_model.pkl                    # Trained CRF model
├── notebooks/
│   └── recipe_ner_solution.ipynb       # Complete solution notebook
├── src/
│   ├── data_preprocessing.py            # Data loading and cleaning
│   ├── feature_engineering.py          # Feature extraction functions
│   ├── model_training.py               # CRF model training
│   └── evaluation.py                   # Model evaluation utilities
├── results/
│   ├── confusion_matrices/             # Performance visualizations
│   └── classification_reports/         # Detailed metrics
├── README.md
└── requirements.txt
```

## Features

### Feature Engineering
- **Lexical Features**: Token text, lemmatization, POS tags, token shape
- **Pattern Recognition**: Regex patterns for quantities and fractions
- **Contextual Features**: Previous/next token information, boundary markers
- **Domain Knowledge**: Predefined keyword sets for culinary units and quantities

### Model Architecture
- **Algorithm**: Conditional Random Fields (CRF)
- **Implementation**: sklearn-crfsuite
- **Optimization**: L-BFGS with L1/L2 regularization
- **Class Balancing**: Inverse frequency weighting for imbalanced labels

## Results

### Confusion Matrix Analysis
The model shows excellent performance with minimal confusion between classes:

| True/Predicted | Ingredient | Quantity | Unit |
|----------------|------------|----------|------|
| Ingredient     | 2107       | 0        | 0    |
| Quantity       | 0          | 409      | 2    |
| Unit           | 0          | 3        | 355  |

### Common Error Patterns
- Primary confusion: Unit ↔ Quantity (5 total errors)
- Root cause: Ambiguous measurement terms in different contexts
- Impact: Minimal effect on overall system performance

## Development

### Training the Model

```python
# Run the complete training pipeline
python src/model_training.py --data data/ingredient_and_quantity.json --output models/crf_model.pkl

# Or use the Jupyter notebook
jupyter notebook notebooks/recipe_ner_solution.ipynb
```

### Evaluation

```python
# Evaluate model performance
python src/evaluation.py --model models/crf_model.pkl --test-data data/test_recipes.json
```

## Dataset

- **Format**: JSON with input/label pairs
- **Size**: 280 recipes (196 training, 84 validation)
- **Labels**: ingredient, quantity, unit
- **Source**: Curated recipe ingredient lists with manual annotations

### Sample Data Format
```json
[
    {
        "input": "2 cups flour 1 teaspoon salt",
        "pos": "quantity unit ingredient quantity unit ingredient"
    }
]
```
