Para executar, basta chamar a função gym.make() e passar como parâmetro da função o ambiente. Em seguida, chamar o método gym.render(nome_do_arquivo), em que o nome do arquivo é um XML gerado pelo software SUPREMICA, tendo a estrutura de uma máquina de estados como conteúdo.

O método env.mapping() informa o nome de todos os eventos da máquina de estados e também seu ID. O método env.step(evento) recebe como parâmetro o ID do evento que possa ser acionado, para então a máquina de estados migrar o seu estado atual. Se o env.step() receber como parâmetro um evento que não possa ser acionado por causa da estrutura do autômato, o usuário será informado que aquela transição é inválida, caso contrário, será acionada a transição.

Por fim, o método env.render() mostra uma imagem com o autômato em questão, tendo o estado atual colorido com amarelo.

O exemplo da pasta env test.py mostra todas as funcões necessárias para fazer o uso do repositório.
