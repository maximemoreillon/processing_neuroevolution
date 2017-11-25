Population my_population;

// Evolution parameters
int population_size = 100;

// Neural networks parameters
int[] network_structure = {10,2};
int network_inputs = 9;

// Training data
Table training_inputs_table;
Table training_outputs_table;

float[][] training_inputs;
float[][] training_outputs;

float[][] validation_inputs;
float[][] validation_outputs;


// Misc parameters
int validation_input_index = 0;


void setup() {
  size(800,600);
  
  // Initialize population of neural networks
  my_population = new Population(population_size, network_structure, network_inputs);
  
  // Initialize training and validation data
  training_data_init();
  validation_inputs = training_inputs;
  validation_outputs = training_outputs;

}


void draw() {
  background(0);
  
  // Compute fitness
  for(int chromosome_index=0; chromosome_index<my_population.chromosomes.length; chromosome_index++) {
    
    float cost = 0.0; // Re-initialize the cost for the new chromosome
    
    // Test chromosome for every training input and output sets
    for (int training_set_index=0; training_set_index<training_inputs.length; training_set_index++) {

      float[] training_input = training_inputs[training_set_index];
      float[] training_output = training_outputs[training_set_index];
      float[] network_output  = my_population.chromosomes[chromosome_index].think(training_input);
      
      for (int output_index=0; output_index<training_output.length; output_index++) {
        cost += 0.5 * (training_output[output_index] - network_output[output_index]) * (training_output[output_index] - network_output[output_index]);
      }
    }
    my_population.chromosomes[chromosome_index].fitness = -cost;
  }
  
  // display the first few networks
  int row_count = 2;
  int column_count = 3;
  for(int row=0; row<row_count; row++){
    for(int column=0; column<column_count; column++){
      
      int chromosome_index = column_count*row+column;
      //float[] network_output = my_population.chromosomes[chromosome_index].think(validation_inputs[validation_input_index]);
      
      
      float pos_x = map(column,-1,column_count,0,width);
      float pos_y = map(row,-1,row_count,0,height);
      float w = width/(column_count+2);
      float h = height/(row_count+2);
      
      my_population.chromosomes[chromosome_index].display(pos_x,pos_y,w,h);
    }
  }
  
  // Apply natural selection
  my_population.natural_selection();
  println("Generation: " + my_population.generation + " mean fitness: " + my_population.mean_fitness + " max fitness: " + my_population.max_fitness + ", min fitness: " + my_population.min_fitness + ", death_toll: " + my_population.death_toll);

  fill(255);
  textSize(24);
  textAlign(LEFT,UP);
  text("Generation: " + my_population.generation,25,50);
}