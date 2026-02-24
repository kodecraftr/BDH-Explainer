import sys

# Define start (Blue) and end (Pink) RGB values
R_START, G_START, B_START = 0, 100, 255     # Bright Blue
R_END, G_END, B_END = 255, 50, 200          # Hot Pink

def print_gradient(text):
    lines = text.splitlines()
    num_lines = len(lines)

    for i, line in enumerate(lines):
        # Calculate the percentage of the way through the text
        ratio = i / max(num_lines - 1, 1)

        # Interpolate the RGB values
        r = int(R_START + (R_END - R_START) * ratio)
        g = int(G_START + (G_END - G_START) * ratio)
        b = int(B_START + (B_END - B_START) * ratio)

        # Print using ANSI true color escape sequences
        print(f"\033[38;2;{r};{g};{b}m{line}\033[0m")

if __name__ == "__main__":
    # Read all text from standard input (pipe)
    input_text = sys.stdin.read()
    print_gradient(input_text)
