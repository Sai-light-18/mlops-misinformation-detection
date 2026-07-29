import json
import os

def main():
    # Load baseline results
    with open('results/baseline_results.json', 'r') as f:
        results = json.load(f)

    # RoBERTa results from your completed training
    results['roberta_base'] = {
        'accuracy'  : 0.6511,
        'precision' : 0.6473,
        'recall'    : 0.6305,
        'f1_macro'  : 0.6294,
        'epochs'    : 3,
        'note'      : 'Best checkpoint at epoch 3'
    }

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON — LIAR BINARY TEST SET")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1 Macro':>9}")
    print("-" * 60)
    for model, metrics in results.items():
        name = model.replace('_', ' ').title()
        print(f"{name:<25} "
              f"{metrics['accuracy']:>9.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f} "
              f"{metrics['f1_macro']:>9.4f}")
    print("=" * 60)

    # Save full results
    os.makedirs('results', exist_ok=True)
    with open('results/full_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Full evaluation saved to results/full_evaluation.json")

if __name__ == '__main__':
    main()
