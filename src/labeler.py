"""
Darija Sentiment Labeling Tool
================================
Simple CLI tool to manually label Darija comments
as Positive / Negative / Neutral.

Usage:
    python labeler.py --input data/raw/hespress_comments_raw.csv
                      --output data/labeled/hespress_labeled.csv
                      --start 0

Controls:
    1 = Positive  (إيجابي)
    2 = Negative  (سلبي)
    3 = Neutral   (محايد)
    s = Skip
    q = Quit & save

Author: Khalid Morjan
Project: Darija Sentiment Analysis
"""

import pandas as pd
import argparse
import os
import sys


LABEL_MAP = {
    "1": "positive",
    "2": "negative", 
    "3": "neutral",
}

COLORS = {
    "positive": "\033[92m",  # Green
    "negative": "\033[91m",  # Red
    "neutral":  "\033[93m",  # Yellow
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "blue":     "\033[94m",
    "cyan":     "\033[96m",
}


def color(text, c):
    return f"{COLORS.get(c, '')}{text}{COLORS['reset']}"


def display_comment(idx, total, row):
    """Display a comment for labeling."""
    print("\n" + "="*60)
    print(color(f"  Comment {idx}/{total}", "blue"))
    print("="*60)
    
    if row.get("category"):
        print(color(f"  Category: {row['category']}", "cyan"))
    if row.get("article_title"):
        title = str(row['article_title'])[:80] + "..." if len(str(row['article_title'])) > 80 else row['article_title']
        print(color(f"  Article: {title}", "cyan"))
    
    print()
    print(color("  " + str(row["text"]), "bold"))
    print()
    print("─"*60)
    print(
        color("  [1] Positive  ", "positive") +
        color("  [2] Negative  ", "negative") +
        color("  [3] Neutral  ", "neutral") +
        "  [s] Skip  [q] Quit"
    )


def run_labeler(input_csv, output_csv, start_idx=0):
    """Interactive labeling session."""
    
    # Load data
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    
    # Use cleaned text if available
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    
    # Filter out already labeled (if resuming)
    if "label" in df.columns:
        unlabeled_mask = df["label"].isna() | (df["label"] == "")
        unlabeled_df = df[unlabeled_mask].copy()
    else:
        df["label"] = ""
        unlabeled_df = df.copy()
    
    total = len(unlabeled_df)
    print(color(f"\n🏷️  Darija Sentiment Labeler", "bold"))
    print(color(f"📊 Comments to label: {total}", "blue"))
    print(color(f"💾 Output: {output_csv}", "cyan"))
    print("\nLabel each comment:")
    print(color("  1 = Positive (مزيان، واو، شكراً...)", "positive"))
    print(color("  2 = Negative (خايب، غلط، مقبولش...)", "negative"))
    print(color("  3 = Neutral (خبر، معلومة، سؤال...)", "neutral"))
    print("  s = Skip (unclear, too short)")
    print("  q = Save and quit\n")
    
    labeled_count = 0
    
    for i, (orig_idx, row) in enumerate(unlabeled_df.iterrows()):
        if i < start_idx:
            continue
        
        display_comment(i + 1, total, row)
        
        while True:
            try:
                choice = input("\n  Your label: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = "q"
            
            if choice == "q":
                # Save and exit
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                print(color(f"\n✅ Saved {labeled_count} labels to {output_csv}", "positive"))
                print(color(f"📊 Progress: {labeled_count}/{total} labeled", "blue"))
                sys.exit(0)
                
            elif choice == "s":
                print(color("  ⏭  Skipped", "neutral"))
                break
                
            elif choice in LABEL_MAP:
                label = LABEL_MAP[choice]
                df.at[orig_idx, "label"] = label
                labeled_count += 1
                
                color_key = label
                print(color(f"  ✓  Labeled as: {label.upper()}", color_key))
                
                # Auto-save every 50 labels
                if labeled_count % 50 == 0:
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                    print(color(f"  💾 Auto-saved ({labeled_count} labels)", "cyan"))
                break
            else:
                print(color("  ❌ Invalid input. Use 1, 2, 3, s, or q", "negative"))
    
    # Final save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    # Summary
    label_counts = df["label"].value_counts()
    print(color(f"\n🎉 Labeling complete!", "positive"))
    print(color(f"📊 Label distribution:", "blue"))
    for label, count in label_counts.items():
        if label:
            print(f"   {label}: {count}")
    print(color(f"💾 Saved to: {output_csv}", "cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darija Sentiment Labeling Tool")
    parser.add_argument("--input", default="data/raw/hespress_comments_raw.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="data/labeled/hespress_labeled.csv",
                        help="Output CSV file")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from comment index N (for resuming)")
    
    args = parser.parse_args()
    run_labeler(args.input, args.output, args.start)
