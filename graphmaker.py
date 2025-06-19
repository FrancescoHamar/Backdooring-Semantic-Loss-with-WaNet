import torch
from torch import tensor
import matplotlib.pyplot as plt

# Example input
asr05cor = [tensor(66.7275), tensor(67.2053), tensor(68.6764), tensor(71.1369), tensor(79.8031), tensor(83.4120), tensor(86.6551), tensor(83.1511), tensor(90.2787), tensor(89.3364), tensor(85.9444), tensor(77.2489), tensor(86.8044), tensor(82.3667), tensor(82.0453), tensor(87.8836), tensor(92.5587), tensor(89.2413), tensor(88.2298), tensor(95.7578), tensor(91.3902), tensor(91.5844), tensor(86.4880), tensor(95.7191), tensor(95.2698), tensor(95.7951), tensor(94.1102), tensor(95.2160), tensor(94.5942), tensor(94.9902)]

clean05cor = [tensor(79.8165), tensor(82.2511), tensor(84.2591), tensor(83.8347), tensor(82.4502), tensor(81.8986), tensor(78.6947), tensor(81.4840), tensor(82.8569), tensor(84.1160), tensor(86.4058), tensor(84.4716), tensor(87.8569), tensor(89.0040), tensor(87.0338), tensor(87.8640), tensor(86.8725), tensor(87.0267), tensor(88.5800), tensor(82.4422), tensor(88.2760), tensor(88.3520), tensor(89.4307), tensor(84.0645), tensor(86.7751), tensor(86.7707), tensor(88.4991), tensor(88.5347), tensor(86.9658), tensor(86.2258)]

asr05heu = [tensor(60.1325), tensor(61.0270), tensor(60.2689), tensor(62.1636), tensor(63.5978), tensor(62.3684), tensor(62.4596), tensor(63.9449), tensor(62.6142), tensor(65.0200), tensor(70.1080), tensor(76.6573), tensor(78.2804), tensor(80.5191), tensor(78.2609), tensor(71.6565), tensor(71.9551), tensor(71.6724), tensor(78.6902), tensor(80.6578), tensor(85.2431), tensor(88.3805), tensor(81.8680), tensor(83.1515), tensor(79.0645), tensor(92.2946), tensor(90.8951), tensor(93.0142), tensor(94.7231), tensor(90.6595)]

clean05heu = [tensor(75.5773), tensor(76.8035), tensor(77.8262), tensor(79.0076), tensor(78.6956), tensor(80.2235), tensor(80.5191), tensor(80.2062), tensor(81.0693), tensor(80.2076), tensor(80.2120), tensor(79.3382), tensor(76.3449), tensor(76.1271), tensor(79.9969), tensor(82.5564), tensor(83.4089), tensor(82.8720), tensor(83.3658), tensor(83.1462), tensor(82.2107), tensor(80.7564), tensor(84.1164), tensor(83.6645), tensor(84.7391), tensor(86.1289), tensor(86.9373), tensor(85.1236), tensor(85.4249), tensor(87.3431)]

asr05impl = [tensor(65.8298), tensor(65.4200), tensor(66.4671), tensor(68.2089), tensor(68.2969), tensor(83.3676), tensor(84.9924), tensor(75.6964), tensor(88.5773), tensor(91.7280), tensor(86.6000), tensor(96.3582), tensor(94.6316), tensor(98.1684), tensor(95.7120), tensor(97.4267), tensor(96.0915), tensor(96.7840), tensor(92.7578), tensor(91.3342), tensor(98.2453), tensor(97.9742), tensor(96.2338), tensor(95.3209), tensor(93.5720), tensor(96.2280), tensor(96.5120), tensor(97.9622), tensor(93.2253), tensor(97.3022)]

clean05impl = [tensor(80.5791), tensor(81.9307), tensor(82.1427), tensor(82.2280), tensor(85.2462), tensor(84.2289), tensor(86.5649), tensor(87.7516), tensor(80.2467), tensor(87.3591), tensor(86.4453), tensor(83.0764), tensor(87.6676), tensor(81.5147), tensor(84.7929), tensor(85.5204), tensor(88.9755), tensor(88.0978), tensor(89.2004), tensor(89.1449), tensor(85.1018), tensor(86.4675), tensor(88.9693), tensor(88.9196), tensor(88.4587), tensor(89.6578), tensor(89.7893), tensor(86.8796), tensor(89.1222), tensor(87.0351)]

asr75cor = [tensor(70.7191), tensor(67.1885), tensor(66.7662), tensor(66.1124), tensor(69.9098), tensor(64.8191), tensor(80.7631), tensor(77.3618), tensor(88.1991), tensor(78.5062), tensor(87.4596), tensor(88.4293), tensor(86.4671), tensor(92.4640), tensor(90.2769), tensor(93.5675), tensor(91.5995), tensor(94.8422), tensor(93.6920), tensor(94.7684), tensor(93.5347), tensor(95.5631), tensor(92.4942), tensor(92.4124), tensor(95.6747), tensor(94.6333), tensor(95.3782), tensor(95.9436), tensor(95.5178), tensor(94.8489)]

clean75cor = [tensor(76.0911), tensor(80.3084), tensor(84.2880), tensor(85.6853), tensor(85.2720), tensor(85.6613), tensor(84.1284), tensor(85.8311), tensor(84.8004), tensor(86.9058), tensor(87.4165), tensor(87.5982), tensor(88.9489), tensor(86.3827), tensor(87.1955), tensor(86.2467), tensor(88.2076), tensor(86.3818), tensor(88.0644), tensor(87.1098), tensor(89.1889), tensor(86.3329), tensor(88.3458), tensor(89.5538), tensor(87.9298), tensor(89.1791), tensor(88.8027), tensor(88.3107), tensor(88.8644), tensor(89.3200)]

asr75heu = [tensor(73.1515), tensor(72.3316), tensor(72.2476), tensor(70.7785), tensor(70.6951), tensor(71.6960), tensor(70.9973), tensor(71.1907), tensor(71.6853), tensor(69.6200), tensor(71.6996), tensor(70.6018), tensor(72.9453), tensor(71.0938), tensor(69.7058), tensor(82.1027), tensor(76.6818), tensor(70.0155), tensor(69.0316), tensor(83.0618), tensor(85.2578), tensor(79.8342), tensor(80.2658), tensor(85.4098), tensor(79.7395), tensor(85.2333), tensor(86.8542), tensor(93.2675), tensor(90.2778), tensor(94.1422)]

clean75heu = [tensor(77.0973), tensor(78.4613), tensor(78.5569), tensor(79.6000), tensor(79.8538), tensor(80.3004), tensor(81.0889), tensor(79.5711), tensor(81.6613), tensor(81.9253), tensor(82.0840), tensor(81.3120), tensor(83.3156), tensor(84.9653), tensor(85.5738), tensor(80.5782), tensor(84.2289), tensor(86.6324), tensor(86.5373), tensor(83.9596), tensor(82.1324), tensor(87.1907), tensor(86.2551), tensor(85.0609), tensor(87.7702), tensor(86.7662), tensor(86.2707), tensor(83.3640), tensor(86.8227), tensor(86.6115)]

asr75impl = [tensor(65.0560), tensor(63.8480), tensor(65.6413), tensor(66.9116), tensor(88.8427), tensor(76.3698), tensor(76.5307), tensor(93.3778), tensor(95.9387), tensor(84.1267), tensor(87.8186), tensor(96.7396), tensor(86.9124), tensor(94.4213), tensor(93.6298), tensor(88.3831), tensor(97.1506), tensor(94.7698), tensor(93.6809), tensor(96.5360), tensor(94.2275), tensor(97.5431), tensor(97.6520), tensor(95.2475), tensor(98.7231), tensor(93.3591), tensor(92.4293), tensor(95.1836), tensor(94.5942), tensor(96.2445)]

clean75impl = [tensor(78.4165), tensor(80.1582), tensor(80.4551), tensor(85.5627), tensor(78.5702), tensor(85.1885), tensor(85.6004), tensor(76.1702), tensor(78.5902), tensor(88.1316), tensor(88.2871), tensor(85.1266), tensor(89.1133), tensor(87.3298), tensor(88.1240), tensor(88.4209), tensor(86.9453), tensor(88.8875), tensor(89.3933), tensor(88.0996), tensor(88.0605), tensor(87.9805), tensor(87.8066), tensor(90.2267), tensor(83.7124), tensor(88.3116), tensor(88.7218), tensor(89.5120), tensor(90.0147), tensor(89.5835)]

asr15cor = [tensor(66.6907), tensor(65.2355), tensor(66.5835), tensor(67.5738), tensor(77.0364), tensor(87.2391), tensor(86.2902), tensor(95.2022), tensor(95.9947), tensor(94.8355), tensor(95.9920), tensor(96.2813), tensor(96.6102), tensor(96.4165), tensor(97.1507), tensor(97.0884), tensor(97.2316), tensor(97.0191), tensor(97.1427), tensor(97.0818), tensor(97.2000), tensor(96.5071), tensor(97.2889), tensor(97.2707), tensor(96.9347), tensor(97.3231), tensor(97.3005), tensor(97.2609), tensor(97.0960), tensor(97.2173)]

clean15cor = [tensor(81.3991), tensor(82.4329), tensor(83.2107), tensor(82.8871), tensor(85.2031), tensor(84.7782), tensor(84.4075), tensor(85.8529), tensor(87.5938), tensor(87.6533), tensor(87.8653), tensor(88.0987), tensor(87.7356), tensor(87.7720), tensor(89.0409), tensor(89.6031), tensor(89.9018), tensor(90.0284), tensor(90.2867), tensor(89.9862), tensor(89.7413), tensor(90.0076), tensor(89.6276), tensor(90.0551), tensor(89.7142), tensor(89.4676), tensor(90.1627), tensor(90.2404), tensor(90.5929), tensor(89.8742)]

asr15heu = [tensor(75.1444), tensor(74.9725), tensor(74.8480), tensor(73.7160), tensor(73.6164), tensor(71.0471), tensor(71.6462), tensor(70.3582), tensor(72.1689), tensor(75.2835), tensor(82.5498), tensor(82.7027), tensor(90.8102), tensor(91.7231), tensor(92.0422), tensor(93.6956), tensor(92.7738), tensor(94.1649), tensor(95.2191), tensor(96.6365), tensor(96.7053), tensor(95.9022), tensor(95.3356), tensor(95.7493), tensor(96.0413), tensor(96.6311), tensor(96.8342), tensor(96.6609), tensor(96.6795), tensor(96.7636)]

clean15heu = [tensor(76.1924), tensor(77.4800), tensor(77.5653), tensor(78.3173), tensor(78.3271), tensor(80.4725), tensor(81.0173), tensor(81.2782), tensor(81.2489), tensor(81.7293), tensor(79.9249), tensor(81.6773), tensor(79.4991), tensor(83.7035), tensor(84.0827), tensor(82.5218), tensor(83.6067), tensor(84.8107), tensor(85.7760), tensor(85.6689), tensor(86.2480), tensor(86.5040), tensor(86.8555), tensor(86.2387), tensor(86.8107), tensor(88.0609), tensor(88.0796), tensor(87.4524), tensor(88.2191), tensor(87.4040)]

asr15impl = [tensor(69.6609), tensor(70.0249), tensor(67.0040), tensor(84.8538), tensor(82.9369), tensor(96.9635), tensor(91.2924), tensor(96.1311), tensor(97.7769), tensor(97.3769), tensor(98.3813), tensor(98.6076), tensor(98.6169), tensor(99.3187), tensor(98.7476), tensor(99.2796), tensor(98.8373), tensor(99.2800), tensor(99.1596), tensor(99.0213), tensor(99.0200), tensor(99.3480), tensor(99.3622), tensor(99.2445), tensor(99.3511), tensor(99.5196), tensor(99.3889), tensor(99.4818), tensor(99.2138), tensor(99.5476)]

clean15impl = [tensor(80.8947), tensor(82.0005), tensor(84.0858), tensor(81.6911), tensor(85.0586), tensor(84.9227), tensor(88.0271), tensor(88.6213), tensor(89.6124), tensor(87.9200), tensor(88.9400), tensor(90.2000), tensor(90.8142), tensor(90.8804), tensor(90.7795), tensor(90.8169), tensor(90.4778), tensor(91.0809), tensor(90.3658), tensor(91.4276), tensor(91.5307), tensor(90.4413), tensor(91.4164), tensor(91.6636), tensor(91.6916), tensor(91.5520), tensor(91.8182), tensor(91.3409), tensor(90.8711), tensor(91.2253)]

asr05nosem = [tensor(65.5911), tensor(67.5267), tensor(79.4662), tensor(83.0391), tensor(87.8436), tensor(84.1169), tensor(79.8316), tensor(93.8320), tensor(89.1880), tensor(84.4378), tensor(93.8418), tensor(97.2284), tensor(97.8467), tensor(93.7182), tensor(98.1831), tensor(98.2938), tensor(95.9796), tensor(89.2538), tensor(96.9769), tensor(97.8404), tensor(96.9498), tensor(96.9786), tensor(97.6880), tensor(96.5004), tensor(96.1747), tensor(98.6373), tensor(97.9733), tensor(97.7369), tensor(94.4298), tensor(97.5338)]

clean05nosem = [tensor(84.1025), tensor(85.5244), tensor(84.6268), tensor(85.1674), tensor(83.5751), tensor(87.4477), tensor(87.7228), tensor(86.0456), tensor(87.7914), tensor(90.7081), tensor(88.7192), tensor(86.8944), tensor(85.6206), tensor(88.0520), tensor(85.6030), tensor(85.6615), tensor(89.2335), tensor(90.1886), tensor(89.4018), tensor(87.6613), tensor(89.4853), tensor(89.5253), tensor(89.7499), tensor(89.6255), tensor(89.7407), tensor(87.9589), tensor(88.7707), tensor(88.8335), tensor(89.7459), tensor(86.9998)]


def compute_average(arrays):
    stacked = torch.stack([torch.tensor([t.item() for t in arr]) for arr in arrays])
    return torch.mean(stacked, dim=0)

def plot_named_arrays(ax, named_arrays, title):
    for name, tensor_list in named_arrays.items():
        values = [t.item() for t in tensor_list]
        ax.plot(values, label=name)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy %")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

asr_named_arrays = {
    "ResNet18 Baseline": compute_average([asr05nosem]),
    "0.5 mag, all constraint sets avg": compute_average([asr05impl, asr05cor, asr05heu]),
}


clean_named_arrays = {
    "ResNet18 Baseline": compute_average([clean05nosem]),
    "0.5 mag, all constraint sets avg": compute_average([clean05impl, clean05cor, clean05heu]),
}

# Plot them
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_named_arrays(axes[0], asr_named_arrays, "ASR Performance Comparison")
plot_named_arrays(axes[1], clean_named_arrays, "Clean Accuracy Performance Comparison")

plt.tight_layout()
plt.savefig("performance_comparison.png")
plt.show()