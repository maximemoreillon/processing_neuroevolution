Population my_population;

int population_size = 100;

int[] network_structure = {10,2};
int network_inputs = 9;


Table training_inputs_table;
Table training_outputs_table;

float[][] training_inputs;
float[][] training_outputs;

float[][] validation_inputs;
float[][] validation_outputs;

int validation_input_index = 0;


void setup() {
  size(displayWidth,displayHeight);
  
  my_population = new Population(population_size, network_structure, network_inputs);
  
  training_data_init();
  validation_inputs = training_inputs;
  validation_outputs = training_outputs;

}


void draw() {
  background(0);
  
  // simulation and display
  for(int chromosome_index=0; chromosome_index<my_population.chromosomes.length; chromosome_index++) {

    float cost = 0.0;
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
  
  my_population.natural_selection();
  println("Generation: " + my_population.generation + " mean fitness: " + my_population.mean_fitness + " max fitness: " + my_population.max_fitness + ", min fitness: " + my_population.min_fitness + ", death_toll: " + my_population.death_toll);

  fill(255);
  textSize(24);
  text("Generation: " + my_population.generation,25,50);
}