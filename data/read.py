import ast


def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into individual tuples (assuming they are on separate lines or identifiable)
    tuple_strings = content.split('\n')

    # Process each tuple string
    tuples = []
    for tuple_str in tuple_strings:
        if tuple_str.strip():  # Check if the line is not empty
            try:
                # Use ast.literal_eval to safely evaluate the string as a tuple
                parsed_tuple = ast.literal_eval(tuple_str)
                if isinstance(parsed_tuple, tuple):
                    tuples.append(parsed_tuple)
                else:
                    print(f"Warning: Parsed content is not a tuple: {parsed_tuple}")
            except Exception as e:
                print(f"Error parsing tuple: {tuple_str} - {e}")

    return tuples


def get_second_values_from_tuples(tuples):
    second_values = [t[1] for t in tuples if len(t) > 1]
    return second_values


def get_first_values_from_tuples(tuples):
    first_values = [t[0] for t in tuples if len(t) > 1]
    return first_values


# Example usage
# file_path = 'processed_docs.txt'  # Replace with the path to your file
# tuples = read_and_process_file(file_path)
# # Extract the first values
# first_values = get_first_values_from_tuples(tuples)
# second_values = get_second_v  alues_from_tuples(tuples)
# Printing tuples
# for t in tuples:
#     print(t)
