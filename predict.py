import torch
import esm
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    # 1. Load ab ag seq
    print('Task start time:',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    agseq, abseq = loadseq()
    dataset = [('agseq', agseq.upper()), ('abseq', abseq.upper())]

    # 2. Represent sequence using esm-2
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    print("Sequence characterization begins", end="")
    # Extract per-residue representations fro each seq
    representations_list = []
    for i in range(len(dataset)):
        data = [dataset[i]]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens,
                            repr_layers=[33],
                            return_contacts=True)
        token_representations = results["representations"][33]
        representations_list.append(token_representations[0][1:-1])
        print('.', end="")
    print('\tDone')

    # 3. predict
    model1 = torch.load(
        'model_ag.pth')
    model2 = torch.load(
        'model_ab.pth')
    model1.eval()
    model2.eval()  # set eval mode

    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # apply model
    input_sequence1 = representations_list[0].to(device)  # ag feature matrix
    input_sequence2 = representations_list[1].to(device)  # ab feature matrix
    print("Model prediction begins", end="")
    output1 = model1(input_sequence1)
    output2 = model2(input_sequence2)

    result = torch.matmul(output1, output2.transpose(0, 1))
    result = torch.sigmoid(result)
    print("\tDone")
    print(result.shape)

    # set threshold
    threshold = 0.5
    result = result.to('cpu')
    result_binary = np.where(result >= threshold, 1, 0)

    print('threshold: ', threshold)

    # Find the coordinate with value 1
    indices = np.where(result_binary == 1)
    coordinates = list(zip(indices[0] + 1, indices[1] + 1))

    # print sites
    print('---------------------------------------------------------------')
    print("predict_sites: ", coordinates)

    # print epitope and paratope
    predict_epitope = sorted(set([x for x, y in coordinates]))
    predict_paratope = sorted(set([y for x, y in coordinates]))
    print('---------------------------------------------------------------')
    print('predict_epitope:  ', predict_epitope)
    print('predict_paratope: ', predict_paratope)

    # 4. visualization
    # plt(result_binary)
    print('Task end time::',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def loadseq():

    parser = argparse.ArgumentParser(description='please set the antigen and antibody sequences.')
    parser.add_argument(
        '--agseq',
        default=
        'MQIPQAPWPVVWAVLQLGWRPGWFLDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDCRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPRPAGQFQTLVVGVVGGLLGSLVLLVWVLAVICSRAARGTIGARRTGQPLKEDPSAVPVFSVDYGELDFQWREKTPEPPVPCVPEQTEYATIVFPSGMGTSSPARRGSADGPRSAQPLRPEDGHCSWPL',
        help='Description of abseq parameter')
    parser.add_argument(
        '--abseq',
        default=
        'evqllesggvlvqpggslrlscaasgftfsnfgmtwvrqapgkglewvsgisgggrdtyfadsvkgrftisrdnskntlylqmnslkgedtavyycvkwgniyfdywgqgtlvtvssastkgpsvfplapcsrstsestaalgclvkdyfpepvtvswnsgaltsgvhtfpavlqssglyslssvvtvpssslgtktytcnvdhkpsntkvdkrveskygppcppcpapeflggpsvflfppkpkdtlmisrtpevtcvvvdvsqedpevqfnwyvdgvevhnaktkpreeqfnstyrvvsvltvlhqdwlngkeykckvsnkglpssiektiskakgqprepqvytlppsqeemtknqvsltclvkgfypsdiavewesngqpennykttppvldsdgsfflysrltvdksrwqegnvfscsvmhealhnhytqkslslslgk',
        help='Description of agseq parameter')

    # Parse command line arguments
    args = parser.parse_args()

    # Access the value of the parameter
    agseq_value = args.agseq
    abseq_value = args.abseq

    return agseq_value, abseq_value


if __name__ == "__main__":
    main()
