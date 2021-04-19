function  [w1, w2, w3, w4] = DeepLearning(w1, w2, w3, w4, input_Image, correct_Output)
alpha =  0.01; % Para controlar a taxa de aprendizagem

N = 5;
for k= 1:N
    reshaped_input_Image = reshape(input_Image(:, :, k), 25, 1); % Conversão da imagem em uma...
    % matriz com uma única coluna e com 25 linhas referente os valores de pixels.
    
    input_of_hidden_layer1 = w1*reshaped_input_Image; % Entrada de sinal para e 1ª camada de nós.
    output_of_hidden_layer1 = ReLU(input_of_hidden_layer1); % Os sinais que entram na camada
    % oculta são processada pela função de ativação RELU.
    
    input_of_hidden_layer2 = w2*output_of_hidden_layer1; % A saída da primeira camada oculta
    % é a entrada da segunda camada oculta e assim por diante.
    output_of_hidden_layer2 = ReLU(input_of_hidden_layer2);

    input_of_hidden_layer3 = w3*output_of_hidden_layer2; 
    output_of_hidden_layer3 = ReLU(input_of_hidden_layer3);
    
    input_of_hidden_node = w4*output_of_hidden_layer3; 
    final_output = Softmax(input_of_hidden_node); % Os sinais de saída da última 
    % camada serão processados pela função de ativação softmax.
    
    correct_Output_transpose = correct_Output(k,:)'; % Transposição da matriz de entrada para 
    % possbilitar o cálculo do erro.
    error = correct_Output_transpose - final_output;
    
    delta = error; % Para aplicar a Regra de Delta. 
    % Esse delta é do algorítimo back propagation.
    
    error_of_hidden_layer3 = w4'*delta;
    delta3 = (input_of_hidden_layer3>0).*error_of_hidden_layer3;
    
    error_of_hidden_layer2 = w3'*delta3;
    delta2 = (input_of_hidden_layer2>0).*error_of_hidden_layer2;
    
        error_of_hidden_layer1 = w2'*delta2;
    delta1 = (input_of_hidden_layer1>0).*error_of_hidden_layer1;
    
    % Ajuste dos pesos nas camadas ocultas que precisam ser atualizados.
    adjustment_of_w4 = alpha*delta*output_of_hidden_layer3'; 
    adjustment_of_w3 = alpha*delta3*output_of_hidden_layer2';
    adjustment_of_w2 = alpha*delta2*output_of_hidden_layer1';
    adjustment_of_w1 = alpha*delta1*reshaped_input_Image';
    
    % Atualização dos pesos os valores de ajuste
    w1 = w1 + adjustment_of_w1;
    w2 = w2 + adjustment_of_w2;
    w3 = w3 + adjustment_of_w3;
    w4 = w4 + adjustment_of_w4;

end

end

