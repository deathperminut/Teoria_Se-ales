##FUNCTIONS
def Lectura_datos():
    Areas = []
    Antenas_viejas=[]
    TipoAntenasNuevas=[]
    ListaAntenas=["a","b","c","d","e"]
    Zonas=int(input(""))
    for i in range(0,Zonas):
            Area=float(input())
            Areas.append(Area)  
            Antenas=int(input())
            Antenas_viejas.append(Antenas)                 
            TipoAntena=input()
            if(TipoAntena in ListaAntenas):
                    TipoAntenasNuevas.append(TipoAntena)
            else:
                    TipoAntenasNuevas.append(0)
    return Zonas,Areas,Antenas_viejas,TipoAntenasNuevas
def SeleccionarAreaAntena(valor):
    Area=None
    if(valor=="a"):
        Area=32600
    elif(valor=="b"):
        Area=31300
    elif(valor=="c"):
        Area=41300
    elif(valor=="d"):
        Area=52200
    else:
        Area=38700
    return Area
def CalcularAntenas(Zonas,Areas,Antenas_viejas,TipoAntenasNuevas):
    Area_Viejas=17200
    AntenasTotales=0
    ListaAntenas=["a","b","c","d","e"]
    lista=[0,0,0,0,0]
    if(Zonas<=0):
        AntenasTotales=0
    else:
        AntenasZona=[]
        for i in range(0,Zonas):
            if(Areas[i]<0):
                AntenasZona.append(0)
            elif(Antenas_viejas[i]<0):
                    AntenasZona.append(0)
            elif(TipoAntenasNuevas[i]==0):
                AntenasZona.append(0)
            else:
                AreaTipoAntena=SeleccionarAreaAntena(TipoAntenasNuevas[i])
                CalculoAntenasNuevas=(Areas[i]-(Antenas_viejas[i]*Area_Viejas))/AreaTipoAntena
                if(CalculoAntenasNuevas<0):
                    CalculoAntenasNuevas=0
                elif(CalculoAntenasNuevas!=int(CalculoAntenasNuevas)):
                    CalculoAntenasNuevas=int(CalculoAntenasNuevas)+1
                AntenasZona.append(CalculoAntenasNuevas)
                for a in range(0,len(lista)):
                    if(ListaAntenas[a]==TipoAntenasNuevas[i]):
                        lista[a]=lista[a]+CalculoAntenasNuevas

        for antena in AntenasZona:
            AntenasTotales=antena+AntenasTotales
    return AntenasTotales,lista
###MAIN
Zonas,Areas,Antenas_viejas,TipoAntenasNuevas=Lectura_datos()
AntenasTotales,ListaAntenas=CalcularAntenas(Zonas,Areas,Antenas_viejas,TipoAntenasNuevas)
if(AntenasTotales==0):
    print(AntenasTotales)
    print("a"+" "+"0.00%")
    print("b"+" "+"0.00%")
    print("c"+" "+"0.00%")
    print("d"+" "+"0.00%")
    print("e"+" "+"0.00%")
else:
    print(AntenasTotales)
    print("a"+" "+"{:.2f}%".format((ListaAntenas[0]/AntenasTotales)*100))
    print("b"+" "+"{:.2f}%".format((ListaAntenas[1]/AntenasTotales)*100))
    print("c"+" "+"{:.2f}%".format((ListaAntenas[2]/AntenasTotales)*100))
    print("d"+" "+"{:.2f}%".format((ListaAntenas[3]/AntenasTotales)*100))
    print("e"+" "+"{:.2f}%".format((ListaAntenas[4]/AntenasTotales)*100))

