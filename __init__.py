import torch, json
from .nuc_model import *
from .nuc_utils import *
import pandas as pd
import importlib.resources as pkg_resources
from . import data
from . import use_model

class TFBind():
    def __init__(self):
        with pkg_resources.open_text(data, "length.txt") as infile:
            js = infile.read()
            dic = json.loads(js)
            dna_length_max = dic['dna_length_max']
            protein_length_max = dic['protein_length_max']
            self.max_length = dic['max_length']
        self.nuc_model = Nuc()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使用 importlib.resources 读取文件内容
        with pkg_resources.open_binary(use_model, "nuc_20.pth") as f:
            self.nuc_model.load_state_dict(torch.load(f))
            self.nuc_model.to(device)

    def is_combind(self, protein, dna):
        result = predict_target(protein, dna, self.nuc_model, self.max_length)
        if result == "能够结合":
            return "Combine"
        elif result == "不能够结合":
            return "Not Combine"


    def get_combin_list(self, protein, dna):
        combin_list = []
        start = 0
        end = 140
        end_location = int((dna.__len__() - 140) / 10) + 2
        if int((dna.__len__() - 140) / 10) < 0:
            end_location = 2
        for i in range(0, end_location):
            if i == 0:
                start = 0
            else:
                start = end - 20
            end = start + 80
            if end > dna.__len__():
                if (dna.__len__() - start) < 10:
                    break
                small_dna = dna[start:dna.__len__()]

                result = self.is_combind(protein, small_dna)
                if result:
                    dict = {
                        "start": start,
                        "end": dna.__len__(),
                        "dna": small_dna
                    }
                    combin_list.append(dict)
                break
            else:
                small_dna = dna[start:end]
                result = self.is_combind(protein, small_dna)
                if result:
                    dict = {
                        "start": start,
                        "end": end,
                        "dna": small_dna
                    }
                    combin_list.append(dict)
        return combin_list


if __name__ == '__main__':
    tfbind = TFBind()
    # protein = input("please input protein:")
    # dna = input("please input dna:")
    protein = "MSSDTSRRDHAAMAVREVLAGDRKVGTVSRSARRRRLELRRVAEDDAAEVARWPAAVSVIGRRREMEDAIFVAAPFLAASKEAAVEGSGVAEEEGKEEDEGFFAVYDGHGGSRVAEACRERMHVVLAEEVRVRRLLQGGGGGADVEDEDRARWKEAMAACFTRVDGEVGGAEEADTGEQTVGSTAVVAVVGPRRIVVANCGDSRAVLSRGGVAVPLSSDHKPDRPDEMERVEAAGGRVINWNGYRILGVLATSRSIGDYYLKPYVIAEPEVTVMDRTDKDEFLILASDGLWDVVSNDVACKIARNCLSGRAASKYPESVSGSTAADAAALLVELAISRGSKDNISVVVVELRRLRSRTTASKENGR"
    dna2000 = "GCGCCGCAATACTTGCCCGGTGAGCGCGGCGATGTCGTCGGTGAGGACGACGTCGGAGTCGAGGTAGACGACGCGGCAGGAGGGGAGCGTGGAGGCGAGGTAGGAGCGGTCGTAGTTGAGCGGGCGGTCGAGCGTGCCGCGGATGGAGGTGAAGATGAGCCCCGCGACGCAGGACTCGTCGAACGGGTACACCCGGAACACCAGCGACGGGAACGACGCCCGGACGGTATCCCTCAGCTCCCTCACCGCCGCCTCCGAAGAGACGGCGAGGAAGTGGAAGTGGACTGACCCCGGGCAGGAGGCATGGCGGAGGACGGAGAGCACCGCGGCTATGGTGCCCCTCAGGTATGACGCGTCCAGCGTCATGGCCACGTGCACCGCCGCGTGCGGCGAGCACGCCATGTCCGCGTCCGTCGCCGGCAGCGGCGGCGCTGCAAAGTTGGTGAACTGCAGCGCCTAGCTCCCTGTACTCCGGCACGGCCACCGCCACCGACGCGCGGCTCGAGCGCGGCGCGGTGGATGACGAGGAACGCGGGCGTCCGCGACCGCATGAGCCCGTGGCATGCGTCCACGACATCGTCGGACGCGGTGGGCCGCCCGCCACAGCAGCCGCTCCGCCACTGCCCGCAGCCGCCGGCTCACCCTAGCCACCACGCACAGCGCCCGCGGTCCCAGTTCAGCGCCCAGAACACCAACTCCAGCACCTTCTCGTCCATGATCCGCGGCGGCCCCAGGCGGCGGAGCGTTGGCCGCAGGCGCGGTGGTCACAGGCGGCGGCGCGTCGGCAGCAGGCGTCGTCCCAAGTGTCGAAACGACGGGCGGCGAGGAGTGGCAGGCGGCCTCCTCCTTTACCTATGGTTTAGGATAGGATTATTTGGAGTGAGGGTAATTTGGTCA"
    print(tfbind.get_combin_list(protein, dna2000))