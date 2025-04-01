import matplotlib.font_manager as fm
import pandas as pd

# Get all fonts
font_list = sorted([f.name for f in fm.fontManager.ttflist])

# Filter for possible Chinese fonts
possible_chinese_fonts = [f for f in font_list if any(keyword in f.lower() for keyword in 
                         ['chinese', 'cjk', 'han', 'song', 'ming', 'hei', 'yuan', '黑', '宋', '圆', '明'])]

# Print results
print("==== All Available Fonts ====")
print(pd.Series(font_list).to_string())
print("\n==== Possible Chinese Fonts ====")
print(pd.Series(possible_chinese_fonts).to_string() if possible_chinese_fonts else "No Chinese fonts detected")
