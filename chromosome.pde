class Chromosome{
  // A neural network chromosome
  
  
  float[][][] synapses;
  
  boolean alive;
  float fitness;
  float mutation_factor = 0.1;
  float initial_variance = 2;
  
  Chromosome(int[] network_structure, int network_input_count){
    // Constructor

    synapses = new float[network_structure.length][][]; // defined as [layer][neuron][input]
    
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {

      int neuron_count = network_structure[layer_index];
      int input_count;
      if (layer_index == 0) {
        // First layer takes the network input as input
        input_count = network_input_count;
      } else {
        // Other layers take the neurons of the previous layer as input
        input_count = network_structure[layer_index-1];
      }

      synapses[layer_index] = new float[neuron_count][input_count+1]; // Adding bias

      // Synapses initialized randomly
      for (int neuron_index=0; neuron_index<synapses[layer_index].length; neuron_index++) {
        for (int input_index=0; input_index<synapses[layer_index][neuron_index].length; input_index++) {
          synapses[layer_index][neuron_index][input_index] = random(-initial_variance, initial_variance);
        }
      }
    }
    
    alive = true;
    fitness = 0;
  }

  void born_from(Chromosome parent) {
    // Chromosome has genes similar to its parents
    
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
      for (int neuron_index=0; neuron_index<synapses[layer_index].length; neuron_index++) {
        for (int input_index=0; input_index<synapses[layer_index][neuron_index].length; input_index++) {
          synapses[layer_index][neuron_index][input_index] = parent.synapses[layer_index][neuron_index][input_index] + mutation_factor * randomGaussian();;
        }
      }
    }
  }
  
  
  float[] think(float[] network_input) {
    // Forward propagation through the network
    
    // the input of the first layer is the network input
    float[] input = network_input;
    
    // for each layer
    for(int layer_index = 0; layer_index < synapses.length; layer_index ++) {
      
      // create a layer of neurons according to the synapses
      float[] neurons = new float[synapses[layer_index].length];
      
      // For each one of those neurons
      for(int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index ++) {
        
        // compute the weighted sum of its inputs
        float synaptic_sum = 0;
        
        // deal first with normal inputs of the layer
        for (int input_index=0; input_index < synapses[layer_index][neuron_index].length-1; input_index++) {
          synaptic_sum += input[input_index]*synapses[layer_index][neuron_index][input_index];
        }

        // deal with bias separately
        int bias_index = synapses[layer_index][neuron_index].length-1;
        synaptic_sum += synapses[layer_index][neuron_index][bias_index];
        
        if(layer_index == synapses.length-1){
          // identity for the last layer
          neurons[neuron_index] = identity(synaptic_sum);
        }
        else {
          neurons[neuron_index] = atan(synaptic_sum);
        }
        
      }
      // inputs of the next layer is the neurons of the current
      input = neurons;
    }
    
    return input;
  }
  
  float sigmoid(float x) {
    return 1.0/(1.0+exp(-x));
  }
  
  float identity(float x) {
    return x;
  }
  
  float softsign(float x){
    return 1.0/(1+abs(x));
  }
  
  float tanh(float x){
    return -1+2/(1+exp(-2*x));
  }
  
  
}