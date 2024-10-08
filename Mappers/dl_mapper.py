from Instructions.Deep_Learning.ANN.neuron import neuron_explanation
from Instructions.Deep_Learning.ANN.perceptron import perceptron_explanation
from Instructions.Deep_Learning.ANN.mlp import mlp_explanation
from Instructions.Deep_Learning.ANN.ann import ann_markdown_text
from Instructions.Deep_Learning.Basic_Concepts.optimizers import optimizers_markdown_code


DL_CONCEPTS_MAPPER = {
    "Optimizers": [
        optimizers_markdown_code,
        [
            "./Instructions/Deep_Learning/Basic_Concepts/sdg.png",
            "./Instructions/Deep_Learning/Basic_Concepts/momentum.png",
            "./Instructions/Deep_Learning/Basic_Concepts/nag.png",
            "./Instructions/Deep_Learning/Basic_Concepts/adagrad.png",
            "./Instructions/Deep_Learning/Basic_Concepts/rmsprop.png",

        ],
        "https://www.youtube.com/results?search_query=Optimizers+in+deep+learning"
    ],

}

ANN_MAPPER = {
    "Neuron": [
        neuron_explanation,
        "./Instructions/Deep_Learning/ANN/neuron.png",
        "https://www.youtube.com/results?search_query=Neuron+in+deep+learning"
    ],
    "Perceptron": [
        perceptron_explanation,
        "./Instructions/Deep_Learning/ANN/perceptron.png",
        "https://www.youtube.com/results?search_query=perceptron+in+deep+learning"
    ],
    "Multi Layer Perceptron (MLP)": [
        mlp_explanation,
        "./Instructions/Deep_Learning/ANN/mlp.png",
        "https://www.youtube.com/results?search_query=MLP+in+deep+learning"
    ],
    "Artificial Neural Networks (ANN)": [
        ann_markdown_text,
        "./Instructions/Deep_Learning/ANN/ann.png",
        "https://www.youtube.com/results?search_query=ANN+in+deep+learning"
    ],
}
