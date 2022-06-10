"""

  FUNCIONES CUANTIZAR_BINARIZAR

"""

def Cuanti_PCM(Funcion_muestreada:np.array,bits:int=3,plot:bool=False):
        """
        Funcion encargada de realizar el proceso de cuantización, es decir limitar los valores discretizados
        asociados a una cantidad de bits.
        -----------------------------------------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------------------------------------
        Funcion_muestreada{np.array}: ARREGLO CON LA SEÑAL DISCRETIZADA
        bits{int}: Entero que representa la cantidad de bits para la cuantización

        Returns
        --------------------------------------------------------------------------
        output_digital{np.array}: señal digital asociada a la cantidad de bits
        estados{np.array}: arreglo con todos los niveles usados para la cuantización
        """
        N_bits=bits ## DEFINIMOS NUMERO DE BITS
        Total_niveles=2**(N_bits) ## NIVELES ASOCIADOS A LA CANTIDAD DE BITS
        resolucion=((np.amax(Funcion_muestreada)-np.amin(Funcion_muestreada))/Total_niveles)  ##DEFINIMOS LA RESOLUCIÓN DEL DIGITALIZADOR
        estados=[] ## DEFINIMOS UN ARREGLO CON POSIBLES ESTADOS
        for i in range(0,Total_niveles):
            estados.append(np.amin(Funcion_muestreada)+resolucion*i)  
        
        output_digital=[] ## DEFINIMOS UN ARREGLO PARA NUESTRA SEÑAL DIGITAL 
        Estados_invertidos=list(reversed(estados)) ## REORGANIZAMOS EL ARREGLO DE ESTADOS
        for valor in Funcion_muestreada:
         if(valor>=estados[Total_niveles-1]):
            output_digital.append(estados[Total_niveles-1])
         else:
          for Estado_Posible in range(1,Total_niveles):
            if(valor>=Estados_invertidos[Estado_Posible]):
             output_digital.append(Estados_invertidos[Estado_Posible])
             break
        return output_digital,estados
def Matrix_bits(n:int):
          """
           Funcion que devuelve una matriz con estados binarios asociado a una cantidad de bits
           ------------------------------------------------------------------------------------
           Parameters
           ------------------------------------------------------------------------------------
           n{int}: entero asociado al numero de bits del cuantizador.

           Returns
           ------------------------------------------------------------------------------------
           Matriz de estados binarios
          """
          ## CODIFICACIÓN (BINARIZACIÓN)
          ## FUNCION PARA GENERAR LAS COMBINACIONES BINARIAS ASOCIADAS AL NUMERO DE BITS DEL SISTEMA 
          if n==0:
              return [[]]
          else:
            m = Matrix_bits(n-1)
            return [[i]+item for i in (0,1) for item in m]

def CheckState(States:np.array,Value:float):
  
  """
   funcion que analiza el estado en el cual se encuentra la señal cuantizada
   -------------------------------------------------------------------------
   Parameter
   ------------------------------------------------------------------------
   States{np.array}: arreglo con todos los estados posibles
   Value{float}: flotante que hace referencia al estado buscado

   Return
   ----------------------------------------------------------------------
   State{int-None}: Valor que indica la posición del estado buscado para el value asociado
                    en caso de no encontrarlo retorna None
   
  """
  State=None
  for i in range(0,len(States)):
    if(Value==States[i]):
      State=i
      break
  return State




