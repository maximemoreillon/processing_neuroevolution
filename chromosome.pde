class Chromosome{
  
  // Class for a neural network chromosome
  
  // Properties of the neural network
  float[][][] synapses; // [layer][neuron][origin]
  
  // Properties related to evolution
  boolean alive;
  float fitness;
  float mutation_factor = 0.01;
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
  
  // Different activation functions for the neural networks
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
  
  // Display function
  void display(float x, float y, float w, float h) {
    // Display the neural network at (x,y) with a width w and height h
    
    // Parameters
    float neuron_radius = 0.05*min(w,h);
    float neuron_stroke_weight = 2;

    float synapse_min_width = 0;
    float synapse_max_width = 4;
    float synapse_min_brightness = 20;
    float synapse_max_brightness = 255;

    // synapses
    stroke(255);
    noFill();
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {

      // find min and max weights of current layer
      float min_weight = 9999;
      float max_weight = -9999;
      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        for (int input_index = 0; input_index < synapses[layer_index][neuron_index].length; input_index++) {
          if (synapses[layer_index][neuron_index][input_index] > max_weight) {
            max_weight = synapses[layer_index][neuron_index][input_index];
          }
          if (synapses[layer_index][neuron_index][input_index] < min_weight) {
            min_weight = synapses[layer_index][neuron_index][input_index];
          }
        }
      }

      float startX = map(layer_index-1, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      float endX = map(layer_index, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      
      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        
        float endY;
        if(layer_index == synapses.length-1) {
           endY = map(neuron_index, -1, synapses[layer_index].length, y-0.5*h, y+0.5*h);
        }
        else {
          endY = map(neuron_index, -1, synapses[layer_index].length+1, y-0.5*h, y+0.5*h);
        }
        
        for (int input_index=0; input_index<synapses[layer_index][neuron_index].length; input_index++) {
          float startY = map(input_index, -1, synapses[layer_index][neuron_index].length, y-0.5*h, y+0.5*h);

          float synapse_weight = synapses[layer_index][neuron_index][input_index];
          float colorMap = 0;
          float widthMap = 0;
          if (synapse_weight>0) {
            colorMap = map(synapse_weight, 0, max_weight, synapse_min_brightness, synapse_max_brightness);
            widthMap = map(synapse_weight, 0, max_weight, synapse_min_width, synapse_max_width);
            stroke(colorMap, 0, 0);
            strokeWeight(widthMap);
          }
          else {
            colorMap = map(synapse_weight, min_weight, 0, synapse_max_brightness, synapse_min_brightness);
            widthMap = map(synapse_weight, min_weight, 0, synapse_max_width, synapse_min_width);
            stroke(colorMap);
            strokeWeight(widthMap);
          }

          line(startX, startY, endX, endY);
        }
      }
    }

    // Neurons
    ellipseMode(RADIUS);
    textAlign(CENTER, CENTER);
    textSize(10);
    strokeWeight(neuron_stroke_weight);
    fill(0);
    
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
      float pos_x = map(layer_index, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      for (int neuron_index = 0; neuron_index < synapses[layer_index].length; neuron_index++) {
        float pos_y;
        if(layer_index == synapses.length-1) {
           pos_y = map(neuron_index, -1, synapses[layer_index].length, y-0.5*h, y+0.5*h);
        }
        else {
          pos_y = map(neuron_index, -1, synapses[layer_index].length+1, y-0.5*h, y+0.5*h);
        }
        stroke(255);
        fill(0);
        ellipse(pos_x, pos_y, neuron_radius, neuron_radius);
      }
    }

    // Bias
    ellipseMode(RADIUS);
    textAlign(CENTER, CENTER);
    strokeWeight(neuron_stroke_weight);
    textSize(10);
    fill(0);
    for (int layer_index=0; layer_index<synapses.length; layer_index++) {
      float pos_y = map(synapses[layer_index][0].length-1, -1, synapses[layer_index][0].length, y-0.5*h, y+0.5*h);  // not too sure about this one
      float pos_x = map(layer_index-1, -1, synapses.length-1, x-0.5*w, x+0.5*w);
      stroke(255, 255, 255);
      fill(0);
      ellipse(pos_x, pos_y, neuron_radius, neuron_radius);
    }
    
    
    
    
    // Inputs
 
    rectMode(CENTER);
    strokeWeight(neuron_stroke_weight);
    fill(0);
    for (int input_index=0; input_index<synapses[0][0].length; input_index++) {
      float pos_x = x-0.5*w;
      float pos_y = map(input_index, -1, synapses[0][0].length+1, y-0.5*h, y+0.5*h);
      stroke(255);
      fill(0);
      rect(pos_x, pos_y, 2*neuron_radius, 2*neuron_radius);
    }
  }
}