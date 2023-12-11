import colorsys

import numpy as np
import pandas as pd

__all__ = ["smiles_importance"]


def float_to_colorhex(value: float, saturation: int = 100, lightness: int = 50) -> str:
    """Convert a float value to a colorhex.

    Args:
        value (float): The float value to convert.
        saturation (int, optional): The saturation value to use in the color conversion (default: 100).
        lightness (int, optional): The lightness value to use in the color conversion (default: 50).

    Returns:
        str: The colorhex corresponding to the input float value.
    """
    hue = value * 120  # Range of hues between red (0) and green (120)
    color = colorsys.hls_to_rgb(hue / 360, lightness / 100, saturation / 100)
    return f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"


def array_float_to_colorhex(array: np.ndarray, saturation: int = 100, lightness: int = 50) -> np.ndarray:
    """Convert a numpy array of float values to an array of colorhexes.

    Args:
        array (np.ndarray): The numpy array of float values to convert.
        saturation (int, optional): The saturation value to use in the color conversion (default: 100).
        lightness (int, optional): The lightness value to use in the color conversion (default: 50).

    Returns:
        np.ndarray: A numpy array of colorhexes corresponding to the input array.
    """
    # Define a vectorized version of the float_to_colorhex() function
    vectorized_func = np.vectorize(float_to_colorhex)

    # Apply the vectorized function to the input array
    return vectorized_func(array, saturation, lightness)


def smiles_importance(smile: str, importance: np.ndarray):
    """Visualize the importance of each token in a SMILES string.

    Args:
        smiles (str): The SMILES string to visualize.
        importance (np.ndarray): The importance of each token in the SMILES string.

    Returns:
        str: The HTML string to display the visualization.
    """
    tokens = [*smile]
    colors = array_float_to_colorhex(importance, 75, 75)

    assert len(tokens) == len(colors), "Importance has to match smile length"

    mark_format = "<mark style='background-color: {0};'>{1}</mark>"
    formats = []
    for token, color in zip(tokens, colors):
        formats.append(mark_format.format(color, token))
    html_text = "".join(formats)
    return html_text


def smiles_set_importance(smiles: list[str], importances: list[np.ndarray]):
    formats = []
    for smile, importance in zip(smiles, importances):
        tokens = [*smile]
        colors = array_float_to_colorhex(importance, 75, 75)

        assert len(tokens) == len(colors), "Importance has to match smile length"

        mark_format = "<mark style='background-color: {0};'>{1}</mark>"
        for token, color in zip(tokens, colors):
            formats.append(mark_format.format(color, token))
        formats.append("<br>")
    html_text = "".join(formats)
    return html_text


class SetImportance:
    def __init__(self) -> None:
        self.df = pd.DataFrame()

    def smile_importance(self, smile: str, importance: np.ndarray) -> str:
        """Visualize the importance of each token in a SMILES string.

        Args:
            smiles (str): The SMILES string to visualize.
            importance (np.ndarray): The importance of each token in the SMILES string.

        Returns:
            str: The HTML string to display the visualization.
        """
        tokens = [*smile]
        colors = array_float_to_colorhex(importance, 75, 75)

        assert len(tokens) == len(colors), "Importance has to match smile length"

        mark_format = "<mark style='background-color: {0};'>{1}</mark>"
        formats = []
        for token, color in zip(tokens, colors):
            formats.append(mark_format.format(color, token))
        html_text = "".join(formats)
        return html_text

    def set_smile_importance(self, smile: str, importance: np.ndarray, index: int, column: str) -> None:
        self.df.loc[index, column] = self.smile_importance(smile, importance)

    def vis_smiles(self) -> str:
        html_text = []
        for i in self.df.index:
            for colname in self.df.columns:
                html_text.append(self.df.loc[i, colname])
                html_text.append("####")
            html_text.append("<br>")
        return "".join(html_text)


# if __name__ == "__main__":
#     from IPython.display import HTML, display

#     smiles = "C1=CC=C(C=C1)C2=CC=CC=C2"
#     importance = np.random.rand(len(smiles))
#     html_text = smiles_importance(smiles, importance)
#     display(HTML(html_text))
