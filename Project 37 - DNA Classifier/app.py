import streamlit as st
import pickle
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter("ignore")

with open("Finalized-Model.pickle", "rb") as file:
    model = pickle.load(file)

encoder = OneHotEncoder(categories=[['a', 'c', 'g', 't']] * 57, handle_unknown='ignore')

Sequence_Dict = {

    "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt": "S10",
    "tgctatcctgacagttgtcacgctgattggtgtcgttacaatctaacgcatcgccaa": "AMPC",
    "gtactagagaactagtgcattagcttatttttttgttatcatgctaaccacccggcg": "AROH",
    "aattgtgatgtgtatcgaagtgtgttgcggagtagatgttagaatactaacaaactc": "DEOP2",
    "tcgataattaactattgacgaaaagctgaaaaccactagaatgcgcctccgtggtag": "LEU1_TRNA",
    "aggggcaaggaggatggaaagaggttgccgtataaagaaactagagtccgtttaggt": "MALEFG",
    "cagggggtggaggatttaagccatctcctgatgacgcatagtcagcccatcatgaat": "MALK",
    "tttctacaaaacacttgatactgtatgagcatacagtataattgcttcaacagaaca": "RECA",
    "cgacttaatatactgcgacaggacgtccgttctgtgtaaatcgcaatgaaatggttt": "RPOB",
    "ttttaaatttcctcttgtcaggccggaataactccctataatgcgccaccactgaca": "RRNAB_P1",
    "gcaaaaataaatgcttgactctgtagcgggaaggcgtattatgcacaccccgcgccg": "RRNAB_P2",
    "cctgaaattcagggttgactctgaaagaggaaagcgtaatatacgccacctcgcgac": "RRNDEX_P2",
    "gatcaaaaaaatacttgtgcaaaaaattgggatccctataatgcgcctccgttgaga": "RRND_P1",
    "ctgcaatttttctattgcggcctgcggagaactccctataatgcgcctccatcgaca": "RRNE_P1",
    "tttatatttttcgcttgtcaggccggaataactccctataatgcgccaccactgaca": "RRNG_P1",
    "aagcaaagaaatgcttgactctgtagcgggaaggcgtattatgcacaccgccgcgcc": "RRNG_P2",
    "atgcatttttccgcttgtcttcctgagccgactccctataatgcgcctccatcgaca": "RRNX_P1",
    "aaacaatttcagaatagacaaaaactctgagtgtaataatgtagcctcgtgtcttgc": "TNAA",
    "tctcaacgtaacactttacagcggcgcgtcatttgatatgatgcgccccgcttcccg": "TYRT",
    "gcaaataatcaatgtggacttttctgccgtgattatagacacttttgttacgcgttt": "ARAC",
    "gacaccatcgaatggcgcaaaacctttcgcggtatggcatgatagcgcccggaagag": "LACI",
    "aaaaacgtcatcgcttgcattagaaaggtttctggccgaccttataaccattaatta": "MALT",
    "tctgaaatgagctgttgacaattaatcatcgaactagttaactagtacgcaagttca": "TRP",
    "accggaagaaaaccgtgacattttaacacgtttgttacaaggtaaaggcgacgccgc": "TRPP2",
    "aaattaaaattttattgacttaggtcactaaatactttaaccaatataggcatagcg": "THR",
    "ttgtcataatcgacttgtaaaccaaattgaaaagatttaggtttacaagtctacacc": "BIOB",
    "catcctcgcaccagtcgacgacggtttacgctttacgtatagtggcgacaatttttt": "FOL",
    "tccagtataatttgttggcataattaagtacgacgagtaaaattacatacctgcccg": "UVRBP1",
    "acagttatccactattcctgtggataaccatgtgtattagagttagaaaacacgagg": "UVRBP3",
    "tgtgcagtttatggttccaaaatcgccttttgctgtatatactcacagcataactgt": "LEXA",
    "ctgttgttcagtttttgagttgtgtataacccctcattctgatcccagcttatacgg": "PORI-L",
    "attacaaaaagtgctttctgaactgaacaaaaaagagtaaagttagtcgcgtagggt": "SPOT42",
    "atgcgcaacgcggggtgacaagggcgcgcaaaccctctatactgcgcgccgaagctg": "M1RNA",
    "taaaaaactaacagttgtcagcctgtcccgcttataagatcatacgccgttatacgt": "GLNS",
    "atgcaattttttagttgcatgaactcgcatgtctccatagaatgcgcgctacttgat": "TUFB",
    "ccttgaaaaagaggttgacgctgcaaggctctatacgcataatgcgccccgcaacgc": "SUBB-E",
    "tcgttgtatatttcttgacaccttttcggcatcgccctaaaattcggcgtcctcata": "STR",
    "ccgtttattttttctacccatatccttgaagcggtgttataatgccgcgccctcgat": "SPC",
    "ttcgcatatttttcttgcaaagttgggttgagctggctagattagccagccaatctt": "RPOA",
    "tgtaaactaatgcctttacgtgggcggtgattttgtctacaatcttacccccacgta": "RPLJ",
    "gatcgcacgatctgtatacttatttgagtaaattaacccacgatcccagccattctt": "PORI-R",
    "aacgcatacggtattttaccttcccagtcaagaaaacttatcttattcccacttttc": "ALAS",
    "ttagcggatcctacctgacgctttttatcgcaactctctactgtttctccatacccg": "ARABAD",
    "gccttctccaaaacgtgttttttgttgttaattcggtgtagacttgtaaacctaaat": "BIOA",
    "cagaaacgttttattcgaacatcgatctcgtcttgtgttagaattctaacatacggt": "DEOP1",
    "cactaatttattccatgtcacacttttcgcatctttgttatgctatggttatttcat": "GALP2",
    "atataaaaaagttcttgctttctaacgtgaaagtggtttaggttaaaagacatcagt": "HIS",
    "caaggtagaatgctttgccttgtcggcctgattaatggcacgatagtcgcatcggat": "HISJ",
    "ggccaaaaaatatcttgtactatttacaaaacctatggtaactctttaggcattcct": "ILVGEDA",
    "taggcaccccaggctttacactttatgcttccggctcgtatgttgtgtggaattgtg": "LACP1",
    "ccatcaaaaaaatattctcaacataaaaaactttgtgtaatacttgtaacgctacat": "LPP",
    "tggggacgtcgttactgatccgcacgtttatgatatgctatcgtactctttagcgag": "TRPR",
    "tcagaaatattatggtgatgaactgtttttttatccagtataatttgttggcataat": "UVRB_P2",
    "atatgaacgttgagactgccgctgagttatcagctgtgaacgacattctggcgtcta": "-867",
    "cgaacgagtcaatcagaccgctttgactctggtattactgtgaacattattcgtctc": "-1169",
    "caaaggcctctaaacgggtcttgaggggttttttgctgaaaggaggaactatatgcg": "-802",
    "ttgacctactacgccagcattttggcggtgtaagctaaccattccggttgactcaat": "-521",
    "cgtctatcggtgaacctccggtatcaacgctggaaggtgacgctaacgcagatgcag": "-918",
    "gccaatttttttcgggcgttgcagtcatttactttagggggatgaaccacttttcttg": "-1481",
    "ctttatgctatttggctcggctttggccacgttttgttgttgtcttcttactcactg": "-518",
    "tcaggttgttcttccactgattccatttggcatctccaggctagagctttcggcttc": "-1596",
    "cgcctttagcagttctttttactccgttgttcctgtcggcctacattacttttgtt": "-1229",
    "tttttcttcctcctctgttggttctagcagtggggcaggctcggacatcttagcggt": "-1360",
    "cctcaaacgttttggatgcttggctatacgcggatgttgattgccagcgtgcacgct": "-1250",
    "ggacttgttgtcactttttagacgctagttttgctgggttacggttcggtctttttt": "-1530",
    "atttatggtaacagcgcagcccttgcgtctggagacggtcgtggatgcgcgatggca": "-1334",
    "gtaagggtcagagcactggtgttgtttttgctgttggttttttgttagttcactgtg": "-1498",
    "atgtgcatgcttcattctctgttttcctgtgacgtgttcgcacggtctcactggcc": "-1586",
    "gttggtacttattgacgttgttgtttgcagtcccttgttttctcctgctgggacttt": "-830",
    "cacttacaaagccttgtctgggagcagacttcttgcttccgacggcatgggctttac": "-1452",
    "tgggatttggctctccaggacttgagacgctcttcgggagggctctcactttacgct": "-1448",
    "cttttttgttatgctttcttgtctgttcactttttgctttactttcgtcttctgggg": "-967",
    "atcctgggcggtttgcgcctgactttgatgttcgccttagagggatgttctactggc": "-1311",
    "tccatgattggcaaacgttctctttctctggttccttccttgttgcgtttgtttctg": "-1551",
    "cgggacttgtttgtgatctgctcttgggcgagttgtttcacggccggcttgagagca": "-963",
    "tttgttgagggtttttctctgctgggactctttgtcgggctttcactttagctcttt": "-969",
    "tggcttacgcctcgcctttttctttcttgccggttcttttcggttgggttgtcctgt": "-948",
    "ttttttggcggtgcctttgcgctcggactttctgttgggggggcgacgccagggctt": "-930",
    "ggggcctgcaggcctttccgtgttctgtttttctttcttctttttctggcttttctt": "-1111",
    "ctagattggcttgtttgttccaaactctgggcacttttcggggggtggttccttttt": "-892",
    "ctttcctggttgttttttcttgggttttggttttttttctttgctggcctttccttt": "-1288",
    "ttttttggcttttctttgttttttggtttctttgggttttgctttcttcttttttt": "-781",
    "tggcttttttcttcttttctttcctttttttctttttcttttctcttttttttcttt": "-1028",
    "cctttttgttggttcttggggttcttcttttctggtgggggtctttttttggctggg": "-831",
    "tgccttttcggttttcttgggggcttgtttggctctggctccttttctctttttttt": "-901",
    "ttttgtttttctttcacttttgggttgggttttttctttgttgttttctttgttggt": "-938",
    "gggtggcttcttcccttgcttctttttctttttttttttggcgggtttctttggggg": "-928",
    "ttcgcctttttgggggctttttttggggtctctttggggggtttctggcctggcttt": "-1145",
    "ctggttggctgcttttcttctgctttgttttgttttttttttctttggttttctttc": "-909",
    "gcctgttttgtttctctttctttggcctgtttcttttctttttctttgttttttttt": "-773",
    "tctttgttttttttcgtctttttttcttttttttctttgcttctttggttgttttct": "-859",
    "cccttttttctttcttttttttggcctttcctttttttgtttttttctttttttgct": "-811",
    "ttgctttgtttcttttcttcttctttttttttctttcttttttttcttttttcttt": "-770",
    "gctgggttgtttctgctttgggggggctgttcttttcttttttcttttttggcctgg": "-1083",
    "atcctgttttggtctgggggtttttttggtttttttttttttctttttctctctggt": "-1331",
    "ttttctttttggggtcttttctttctttttttttctttgttttgctggggctttggg": "-769",
    "cggctttctttttgggttgctctttttttttcttttttggtctttggttttttcttt": "-907",
    "ttttttggggtgggttgctttttctttctttttttttttctctctttctttttcttt": "-1211",
    "ctgctttttggttttttctttcttctttttttttttcttctttttttcttttctttc": "-1475",
    "tggttctgggtttggttttggggggttttttttttctttgttttttttcttttcttt": "-894",
    "cttcttctttttttgcttttcttcttcttcttttttttttttctttctttttttttg": "-747",
    "tctttcttctttggtttctctttggggtttttttttttctttctttttttttttttct": "-963",
    "ttttttctttcttctttttttcttcttggctcttctttcttctttgggctttctttt": "-1503",
    "cttttttcttttttctttctttttttttttttttttttcttcttttttttttttttt": "-1187",
    "tctttttttctttttttctttttttttttttttttttttttttctttttttttttct": "-1292",
    "gttttggcctttctttctcttcttttctttctttttttttctttttttttttttcttt": "-795",
    "ttcttttttcttcttttttcttttttttttttttttttttctttttttctttttttg": "-1131",
    "tttttttctttttttttttttttttttttttttcttttttcttttttttttttcttt": "-853",
    "tttttctttttttttttcttttttttttttttcttttcttttctttttttcttttttt": "-1117",
    "ttcttttttttttcttttttttctttttttttttttttttcttttttttttttttttt": "-825",
    "tttttttttttctttttttttttttttctttttttttttttctttttttttttttttt": "-947",
    "ttttcttttttttttttttttttctttttttttttttttttttttttttttttcttttt": "-1283",
    "ttttttttttttttcttttttttttttttttttttttttttttttttttttttttttt": "-1390",
    "ttttttcttttttttttttttttttttttttttttttttttttttttttttttttttt": "-768",
    "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-1143",
    "ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-748",
    "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-1142",
    "ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-1344",
    "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-1535",
    "ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt": "-1179"
}

st.title("DNA Classifier")

sequence_input = st.text_area("**Enter DNA Sequence:**", "")


def preprocess_sequence(sequence):
    nucleotides = list(sequence)
    nucleotides = [char for char in nucleotides if char in ['a', 'c', 'g', 't']]
    
    if len(nucleotides) != 57:
        st.error("**The Sequence Length Should Be Exactly 57 Nucleotides.**")
        return None
    
    df_sequence = pd.DataFrame([nucleotides])
    encoded_sequence = encoder.fit_transform(df_sequence).toarray()
    
    return encoded_sequence


def preprocess_id(sequence):
    return Sequence_Dict.get(sequence, 'Unknown ID')


def predict_ecoli(sequence):
    preprocessed_sequence = preprocess_sequence(sequence)
    if preprocessed_sequence is None:
        return None, None
    
    prediction = model.predict(preprocessed_sequence)
    
    id_display = preprocess_id(sequence)
    
    return prediction, id_display


if st.button("Classify"):
    result, id_display = predict_ecoli(sequence_input)
    if result is not None:
        if result[0] == 1:
            st.success("**The DNA Sequence Indicates The Presence Of EC.Coli (+).**")
        else:
            st.success("**The DNA Sequence Indicates The Absence Of EC.Coli (-).**")
        
        st.info(f"ID : **{id_display}**")

