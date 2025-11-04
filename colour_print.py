# Function to convert color name to ANSI color code
def color_code(color_name):
    """Convert a color name to its ANSI color code"""
    color_map = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
        'reset': 0
    }
    
    bg_prefix = ''
    if color_name.startswith('bg_'):
        bg_prefix = '4'  # Background colors start with 4 instead of 3
        color_name = color_name[3:]  # Remove 'bg_' prefix
    else:
        bg_prefix = '3'  # Foreground colors start with 3
        
    color_code = color_map.get(color_name.lower())
    if color_code is None:
        return None
    
    return f"{bg_prefix}{color_code - 30}"  # Convert to correct code format

# Example using our custom function
def cprint(text, color_name):
    """Print text with color specified by name"""
    code = color_code(color_name)
    if code:
        print(f"\033[{code}m{text}\033[0m")
    else:
        print(text)  # Fallback to normal print if color not found