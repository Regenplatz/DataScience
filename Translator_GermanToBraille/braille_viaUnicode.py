

dict_brailleGerman6 = {
    "a": "\u2801", "b": "\u2803", "c": "\u2809", "d": "\u2819",
    "e": "\u2811", "f": "\u280B", "g": "\u281B", "h": "\u2813",
    "i": "\u280A", "j": "\u281A", "k": "\u2805", "l": "\u2807",
    "m": "\u280D", "n": "\u281D", "o": "\u2815", "p": "\u280F",
    "q": "\u281F", "r": "\u2817", "s": "\u280E", "t": "\u281E",
    "u": "\u2825", "v": "\u2827", "w": "\u283A", "x": "\u282D",
    "y": "\u283D", "z": "\u2835", " ": "\u2800",
    "ä": "\u281C", "ö": "\u282A", "ü": "\u2833", "au": "\u2821",
    "äu": "\u280C", "ei": "\u2829", "ie": "\u282C", "eu": "\u2823",
    "ch": "\u2839", "sch": "\u2831", "st": "\u283E", "β": "\u282E",
    "#": "\u283C", "1": "\u2801", "2": "\u2803", "3": "\u2809",
    "4": "\u2819", "5": "\u2811", "6": "\u280B", "7": "\u281B",
    "8": "\u2813", "9": "\u280A", "0": "\u281A",
    ".": "\u2832", ",": "\u2802", ";": "\u2806", ":": "\u2812",
    "?": "\u2822", "!": "\u2816", "(": "\u2836", ")": "\u2836",
    '„': "\u2826", '“': "\u2834", '"': "\u2834", "*": "\u2814",
    "_": "\u2824", "'": "\u2804", "capital": "\u2820", "/": "\u280C"
}


def translateToBraille(word):
    word_braille = ""
    str_helper = ""
    double_sounds = ["ch", "ei", "eu", "ie", "äu", "au", "st"]

    i = 0
    while i < len(word):
        if word[i].isupper():
            word_braille += dict_brailleGerman6["capital"]
        str_helper += word[i].lower()

        ## 1 letter word
        if len(word) == 1:
            word_braille += dict_brailleGerman6[word.lower()]

        ## 2 letter word or (length of str_helper=2 and contains last elements of word)
        elif (len(word) == 2) or ((len(str_helper) == 2) and (i > len(word) - 2)):
            ## if double sound
            if str_helper in double_sounds:
                word_braille += dict_brailleGerman6[str_helper]
            else:
                word_braille += dict_brailleGerman6[str_helper[0]]
                word_braille += dict_brailleGerman6[str_helper[1]]
            break

        ##
        elif (i >= 2) and (len(str_helper) >= 3):
            str_helper = str_helper[-3:]

            ## if triple sound in helper
            if str_helper == "sch":
                word_braille += dict_brailleGerman6["sch"]
                ## if end of word, break. Else empty str_helper and move i
                if i == len(word) - 1:
                    break
                else:
                    str_helper = ""

            ## if double sound in the beginning of helper
            elif str_helper[:2] in double_sounds:
                word_braille += dict_brailleGerman6[str_helper[:2]]
                str_helper = str_helper[-1]

            else:
                word_braille += dict_brailleGerman6[word[i - 2].lower()]

                ## cover last 2 characters in word
                if i == len(word) - 1:
                    str_lastTwoChars = word[len(word) - 2:]

                    ## if last 2 characters are double sound
                    if str_lastTwoChars in double_sounds:
                        word_braille += dict_brailleGerman6[str_lastTwoChars]

                    ## no double sound
                    else:
                        word_braille += dict_brailleGerman6[word[i - 1].lower()]
                        word_braille += dict_brailleGerman6[word[i].lower()]

        i += 1
    return word_braille


def main():

    word = "Endlich schön!"
    word_braille = translateToBraille(word)
    print(word_braille)


if __name__ == "__main__":
    main()
