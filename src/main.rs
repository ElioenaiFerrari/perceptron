#[derive(Debug)]
struct Perceptron {
    weights: Vec<Vec<f64>>, // Pesos para cada classe
    biases: Vec<f64>,       // Biases para cada classe
}

impl Perceptron {
    // Inicializa um novo Perceptron com pesos e biases zerados
    fn new(n_inputs: usize, n_classes: usize) -> Perceptron {
        Perceptron {
            // Labels e classes
            weights: vec![vec![0.0; n_inputs]; n_classes], // Inicializa os pesos como 0.0
            // Bias é um valor que é adicionado antes de passar pela função de ativação
            biases: vec![0.0; n_classes], // Inicializa os biases como 0.0
        }
    }

    // Calcula a soma ponderada dos inputs para cada classe
    fn general_fn(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = vec![0.0; self.biases.len()]; // Inicializa as saídas com 0.0
        for class in 0..self.weights.len() {
            let mut sum = 0.0;
            for i in 0..self.weights[class].len() {
                sum += self.weights[class][i] * inputs[i]; // Calcula a soma ponderada dos inputs
            }
            outputs[class] = sum + self.biases[class]; // Adiciona o bias à soma ponderada
        }
        outputs
    }

    // Treina o Perceptron ajustando os pesos e biases com base nos erros
    fn train(&mut self, inputs: Vec<f64>, targets: Vec<f64>, lr: f64) {
        let outputs = self.general_fn(&inputs); // Calcula a saída atual
        let mut errors = vec![0.0; targets.len()]; // Inicializa o vetor de erros com 0.0

        // Calcula o erro como a diferença entre o alvo e a saída
        for i in 0..targets.len() {
            errors[i] = targets[i] - outputs[i];
        }

        // Atualiza os pesos e biases para cada classe
        for class in 0..self.weights.len() {
            for i in 0..self.weights[class].len() {
                self.weights[class][i] += lr * errors[class] * inputs[i]; // Ajusta o peso baseado no erro
            }
            self.biases[class] += lr * errors[class]; // Ajusta o bias baseado no erro
        }
    }

    // Prediz a classe para os inputs fornecidos usando softmax para normalizar as saídas
    fn predict(&self, inputs: Vec<f64>) -> Vec<f64> {
        let outputs = self.general_fn(&inputs); // Calcula a soma ponderada para cada classe
        softmax(outputs) // Normaliza as saídas para probabilidades
    }
}

// Função softmax para normalizar as saídas para probabilidades
fn softmax(outputs: Vec<f64>) -> Vec<f64> {
    let mut exp_outputs = vec![0.0; outputs.len()]; // Inicializa o vetor de exponenciais
    let mut sum_exp = 0.0; // Inicializa a soma das exponenciais

    // Calcula a exponencial de cada saída
    for i in 0..outputs.len() {
        exp_outputs[i] = outputs[i].exp();
        sum_exp += exp_outputs[i]; // Soma todas as exponenciais
    }

    // Divide cada exponencial pela soma para obter probabilidades
    let mut probabilities = vec![0.0; outputs.len()];
    for i in 0..exp_outputs.len() {
        probabilities[i] = exp_outputs[i] / sum_exp;
    }

    probabilities
}

// Função para imprimir a classe prevista com base nas probabilidades
fn print_prediction(prediction: Vec<f64>, classes: Vec<&str>) {
    let mut max_value = prediction[0];
    let mut max_index = 0;

    // Encontra o índice da maior probabilidade
    for i in 1..prediction.len() {
        if prediction[i] > max_value {
            max_value = prediction[i];
            max_index = i;
        }
    }

    // Imprime a classe correspondente à maior probabilidade
    println!("Prediction: {:?} ({})", prediction, classes[max_index]);
}

// use lib csv to read iris.csv
fn load_iris() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rdr = csv::Reader::from_path("iris.csv").unwrap();
    let mut inputs = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let input: Vec<f64> = record
            .iter()
            .take(4)
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        let mut target = vec![0.0; 3];
        let class = record.iter().last().unwrap();
        match class.as_ref() {
            "Setosa" => target[0] = 1.0,
            "Versicolor" => target[1] = 1.0,
            "Virginica" => target[2] = 1.0,
            _ => panic!("Invalid class"),
        }
        inputs.push((input.to_vec(), target));
    }
    inputs
}

fn main() {
    let iris = load_iris();
    // Cria um Perceptron com 3 inputs e 3 classes
    let mut perceptron = Perceptron::new(4, 3);
    let classes = vec!["Setosa", "Versicolor", "Virginica"];

    // Classes para exibição dos resultados

    // Treina o Perceptron por um número de épocas
    let epochs = 10000;
    for _ in 0..epochs {
        for (input, target) in &iris {
            perceptron.train(input.clone(), target.clone(), 0.0001); // Treina o Perceptron
        }
    }

    let setosa = vec![5.1, 3.5, 1.4, 0.2];
    let versicolor = vec![5.6, 2.7, 4.2, 1.3];
    let virginica = vec![6.3, 3.3, 6.0, 2.5];

    // Prediz a classe para os inputs fornecidos
    let prediction = perceptron.predict(setosa.clone());
    print_prediction(prediction, classes.clone());
}
