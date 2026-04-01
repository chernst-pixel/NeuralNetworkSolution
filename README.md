Dieses Projekt implementiert ein vollständig verbundenes (fully connected) künstliches neuronales Netzwerk von Grund auf in C#, ohne externe Machine‑Learning‑Libraries.  
Das Modell wurde auf dem MNIST‑Datensatz trainiert.  

Architektur  
Input Layer : 784 Neuronen (28x28 Pixel)  
Hidden Layer 1: 128 Neuronen, Aktivierung: Relu  
Hidden Layer 2: 64 Neuronen, Aktivierung: Relu  
Output Layer : 10 Neuronen, Aktivierung: Softmax  

Ergebnisse nach 5 Epochen  
Genauigkeit: 96,99%  
Korrekte Vorhersagen: 9 699  
Falsche Vorhersagen: 301  
Durchschnittliche Sicherheit: 98,12%  
