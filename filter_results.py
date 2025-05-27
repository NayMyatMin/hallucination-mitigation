import re
import argparse
import sys
import collections # Import collections for defaultdict

# Define patterns for 'I don't know' type answers (case-insensitive)
# Added more comprehensive patterns
ABSTENTION_PATTERNS = [
    re.compile(r"i (don'?t|do not) know", re.IGNORECASE),
    re.compile(r"i (don'?t|do not) remember", re.IGNORECASE),
    re.compile(r"i (don'?t|do not) have enough information", re.IGNORECASE),
    re.compile(r"i am unable to answer", re.IGNORECASE),
    re.compile(r"i lack the information", re.IGNORECASE),
    re.compile(r"no information provided", re.IGNORECASE),
    re.compile(r"insufficient information", re.IGNORECASE),
    re.compile(r"cannot provide an answer", re.IGNORECASE),
    re.compile(r"unable to determine", re.IGNORECASE)
]

def filter_and_renumber_results(input_filepath, output_filepath):
    '''
    Reads a results file, filters out entries where Ground Truth is 'unknown'
    or Generated Answer indicates inability to answer,
    writes the remaining entries to a new file with renumbered examples,
    recalculates summary statistics based on the filtered data,
    and reports counts of filtered entries.
    '''
    try:
        # Initialize statistics collectors
        hallucination_counts = collections.defaultdict(int)
        total_processed_examples = 0 # Count all examples encountered
        total_valid_examples = 0 # Count examples used for stats
        unknown_gt_count = 0     # Count examples filtered due to 'unknown' GT
        refusal_count = 0        # Count examples filtered due to refusal answer
        scores = {
            "Hallucination Severity": 0.0,
            "Factual Accuracy": 0.0,
            "Overconfidence": 0.0,
            "Overall Reliability": 0.0
        }
        score_keys_map = { # Map file format keys to our dict keys
            "Hallucination Severity": "Hallucination Severity",
            "Factual Accuracy": "Factual Accuracy",
            "Overconfidence": "Overconfidence",
            "Overall Reliability": "Overall Reliability"
        }

        with open(input_filepath, 'r', encoding='utf-8') as infile:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:

                current_example_lines = []
                current_stats = {} # Store stats for the current block
                new_example_number = 0
                is_valid_entry = True # Assume valid until proven otherwise
                inside_gpt_eval_block = False
                was_refusal = False      # Flag for current block refusal
                was_unknown_gt = False   # Flag for current block unknown GT

                for line in infile:
                    # Check if a new example block starts
                    if line.startswith("Example "):
                        # Process the previous block first
                        if current_example_lines:
                            total_processed_examples += 1
                            if is_valid_entry:
                                new_example_number += 1
                                total_valid_examples += 1
                                # Modify the first line to have the new example number
                                current_example_lines[0] = f"Example {new_example_number}\n"
                                outfile.writelines(current_example_lines)
                                # Accumulate stats from the valid processed block
                                if current_stats.get("type"):
                                    hallucination_counts[current_stats["type"]] += 1
                                for key, value in current_stats.get("scores", {}).items():
                                    if key in scores: # Ensure we only add known score types
                                        scores[key] += value
                            else: # Entry was invalid
                                if was_unknown_gt: # Prioritize unknown GT as the reason
                                    unknown_gt_count += 1
                                elif was_refusal: # Otherwise, count as refusal
                                    refusal_count += 1
                                # If neither flag is set but invalid, it's an edge case (ignore for now)

                        # Start a new block
                        current_example_lines = [line]
                        current_stats = {} # Reset stats for the new block
                        is_valid_entry = True # Reset validity
                        inside_gpt_eval_block = False
                        was_refusal = False      # Reset refusal flag
                        was_unknown_gt = False   # Reset unknown GT flag
                    
                    elif current_example_lines: # If we are inside a block
                        current_example_lines.append(line)
                        
                        # --- Check for Generated Answer ---
                        if line.strip().startswith("Generated Answer:"):
                            gen_answer = line.split(":", 1)[1].strip()
                            for pattern in ABSTENTION_PATTERNS:
                                if pattern.search(gen_answer):
                                    is_valid_entry = False # Mark invalid
                                    was_refusal = True     # Set refusal flag
                                    break
                        # --- End of Generated Answer check ---

                        # --- Check for the Ground Truth line ---
                        if line.strip().startswith("Ground Truth:"):
                            gt_value = line.split(":", 1)[1].strip()
                            if gt_value.lower() == 'unknown':
                                is_valid_entry = False  # Mark invalid
                                was_unknown_gt = True   # Set unknown GT flag
                        # --- End of Ground Truth check ---
                        
                        # Check if we entered the GPT evaluation block
                        if line.strip() == "GPT-4o-mini Hallucination Evaluation:":
                             inside_gpt_eval_block = True
                        
                        # Extract stats if inside the block and the entry is potentially valid
                        if inside_gpt_eval_block and is_valid_entry:
                            stripped_line = line.strip()
                            if stripped_line.startswith("Hallucination Type:"):
                                current_stats["type"] = stripped_line.split(":", 1)[1].strip()
                            else:
                                # Check for score lines (e.g., "Hallucination Severity: 9.0/10")
                                for key, dict_key in score_keys_map.items():
                                    if stripped_line.startswith(key + ":"):
                                        try:
                                            score_part = stripped_line.split(":", 1)[1].strip()
                                            score_value = float(score_part.split('/')[0]) # Extract the number before '/'
                                            if "scores" not in current_stats:
                                                current_stats["scores"] = {}
                                            current_stats["scores"][dict_key] = score_value
                                        except (ValueError, IndexError):
                                            print(f"Warning: Could not parse score from line: {line.strip()}", file=sys.stderr)
                                        break # Move to next line once a score is found

                # Process the very last block in the file
                if current_example_lines:
                    total_processed_examples += 1 # Count the last one
                    if is_valid_entry:
                        new_example_number += 1
                        total_valid_examples += 1
                        current_example_lines[0] = f"Example {new_example_number}\n"
                        outfile.writelines(current_example_lines)
                        # Accumulate stats from the last valid block
                        if current_stats.get("type"):
                            hallucination_counts[current_stats["type"]] += 1
                        for key, value in current_stats.get("scores", {}).items():
                             if key in scores:
                                 scores[key] += value
                    else: # Last entry was invalid
                        if was_unknown_gt:
                            unknown_gt_count += 1
                        elif was_refusal:
                            refusal_count += 1
                
                # --- Append the new summary statistics --- 
                outfile.write("\n" + "="*50 + "\n")
                outfile.write("Recalculated Hallucination Analysis (Filtered Data):\n")
                # Add new counting stats
                outfile.write(f"Total examples processed: {total_processed_examples}\n")
                outfile.write(f"Filtered due to 'unknown' Ground Truth: {unknown_gt_count}\n")
                outfile.write(f"Filtered due to Refusal/Abstention: {refusal_count}\n")
                outfile.write(f"Total valid examples evaluated: {total_valid_examples}\n")
                outfile.write("\nHallucination type distribution (based on valid examples):\n")
                if total_valid_examples > 0:
                    for h_type, count in sorted(hallucination_counts.items()):
                        percentage = (count / total_valid_examples) * 100
                        outfile.write(f"  {h_type}: {count} ({percentage:.1f}%)\n")
                else:
                     outfile.write("  No valid examples found to calculate average metrics.\n")
                outfile.write("\nAverage metrics across filtered dataset:\n")
                if total_valid_examples > 0:
                    for key, total_score in scores.items():
                        average_score = total_score / total_valid_examples
                        outfile.write(f"  {key}: {average_score:.2f}/10\n")
                else:
                     outfile.write("  No valid examples found.\n")
                outfile.write("="*50 + "\n")
                # --- End of summary --- 

        print(f"Successfully filtered results and recalculated statistics.")
        print(f"Total examples processed: {total_processed_examples}")
        print(f"Filtered due to 'unknown' Ground Truth: {unknown_gt_count}")
        print(f"Filtered due to Refusal/Abstention: {refusal_count}")
        print(f"Valid entries written to output: {total_valid_examples}") # Use the accurate count
        print(f"Output saved to: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter evaluation results file, remove 'Ground Truth: unknown' or refusal entries, renumber examples, recalculate summary stats, and report filter counts.") # Updated description
    parser.add_argument("input_file", help="Path to the input results file (e.g., result_coqa_consistency.txt)")
    parser.add_argument("output_file", help="Path to save the filtered results file (e.g., result_coqa_consistency_filtered.txt)")
    
    args = parser.parse_args()
    
    filter_and_renumber_results(args.input_file, args.output_file) 