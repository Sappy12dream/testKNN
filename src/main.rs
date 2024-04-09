use std::collections::HashMap;

// Define a struct to hold a data point
#[derive(Debug)]
struct DataPoint {
    features: Vec<f64>,
    label: String,
}

// Define a struct for the KNN classifier
struct KNNClassifier {
    k: usize,
    data: Vec<DataPoint>,
}

impl KNNClassifier {
    // Constructor
    fn new(k: usize) -> KNNClassifier {
        KNNClassifier {
            k,
            data: Vec::new(),
        }
    }

    // Add data points to the classifier
    fn add_data(&mut self, features: Vec<f64>, label: &str) {
        self.data.push(DataPoint {
            features,
            label: label.to_string(),
        });
    }

    // Predict the class of a new data point
    fn predict(&self, features: Vec<f64>) -> String {
        // Calculate distances to all data points
        let distances: Vec<(usize, f64)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, dp)| (idx, euclidean_distance(&dp.features, &features)))
            .collect();

        // Sort distances and take the k nearest neighbors
        let mut sorted_distances = distances.clone();
        sorted_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest_neighbors = &sorted_distances[..self.k];

        // Count the occurrences of each class among the nearest neighbors
        let mut class_counts = HashMap::new();
        for (idx, _) in nearest_neighbors {
            let label = &self.data[*idx].label;
            let count = class_counts.entry(label).or_insert(0);
            *count += 1;
        }

        // Return the class with the highest count
        class_counts
            .iter()
            .max_by_key(|&(_, count)| *count)
            .map(|(label, _)| label.to_string())
            .unwrap()
    }
}

// Function to calculate Euclidean distance between two points
fn euclidean_distance(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() {
    // Create a KNN classifier with k=3
    let mut classifier = KNNClassifier::new(3);

    // Add some training data
    classifier.add_data(vec![1.0, 2.0], "A");
    classifier.add_data(vec![2.0, 3.0], "A");
    classifier.add_data(vec![3.0, 4.0], "B");
    classifier.add_data(vec![4.0, 5.0], "B");

    // Get input from the user for the features of the new data point
    let mut input_features = Vec::new();
    println!("Enter the features of the new data point:");
    for i in 0..2 {
        println!("Feature {}: ", i + 1);
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let feature: f64 = input.trim().parse().expect("Invalid input");
        input_features.push(feature);
    }

    // Predict the class of the new data point
    let prediction = classifier.predict(input_features);
    println!("Prediction: {}", prediction);
}
