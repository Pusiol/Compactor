# Compactor

Aqui demonstramos apenas para fins didáticos um compactador de imagem baseado na Transformada de Wavelet, aos moldes do padrão jpeg 2000.

Funcionamento

Primeiramente a imagem é convertida para o formato iluminância-crominâncias. A seguir, o procedimento consiste em aplicar a transformada de wavelets por duas vezes sobre a imagem, usando a Daubechies 9, para cada um dos três canais. Pela propriedade da transformação, as componentes de detalhe tem muito menos entropia do que a imagem original.

Depois disso a transformada é normalizada e os coeficientes são arredondados para permitir a posterior codificação entrópica. Os canais de crominância podem passar por arredondamento bem mais agressivo, porque não são tão importantes para a percepção da imagem.

Seguindo, os dados são serializados junto com informações para a posterior descompactação. Para o codificador entrópico, foi usado o próprio mecanismo de salvar dados compactados do numpy. Reside aí uma grande potencial área para aprimoramento, mas pela simplicidade isso não é explorado.

No caminho de volta, os dados são recuperados, desserializados, desnormalizados e é aplicada a transformada inversa para remontar a imagem original.

Considerações

Isto tem por fim ser apenas uma criação didática, muito distante de um compactador comercial. O usuário deve garantir que as imagens de entrada tenham largura e altura múltiplas de 4. Ao chamar o compactador, deve ser passado o nome da imagem, e ele gerará o arquivo npz, que pode ser lido pelo descompactador.

A taxa de compactação deve ser comparada com o bitmap puro. De fato outros padrões, como os universais jpeg e png, fazem um excelente trabalho, não muito simples de ser superado. Há certo overhead no procedimento adotado também, o que faz com que o método seja bastante fraco com imagens pequenas.

