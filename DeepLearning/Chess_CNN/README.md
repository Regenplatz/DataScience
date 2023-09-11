## CHESS

For this project, data from [kaggle]() was used. This data consists of images from chessboards with the *Forsyth-Edwards-Notation* (FEN) as filenames.

Running [main.py](main.py) processes all steps that are explained below:

- [prepareData.py](prepareData.py) contains functions to load the data and to extract FEN from filenames.
It furthermore contains functions to preprocess images in terms of *rescaling* and *demeaning*.

- FEN is translated to array (8x8) with letters, later with binary one-hot encodings to symbolize the chessman occupying the respective field via [FENfromFilename.py](FENfromFilename.py)

- Each chess board image is cut into 64 field images and saved with information of the corresponding chessman occupying this field (or if empty as empty field) in [fieldOccupation.py](fieldOccupation.py)

- Chessmen and empty fields are finally trained by this new created dataset in [CNN.py](CNN.py). Quality of recognition is tracked via *categorical accuracy*.
