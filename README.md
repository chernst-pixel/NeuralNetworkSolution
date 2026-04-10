# Neural Network Solution  

Dieses Projekt ist eine eigenständige Implementierung eines neuronalen Netzes mit dem Ziel,  
die grundlegenden Konzepte des Machine Learnings praxisnah zu erlernen und besser zu verstehen.

## Projektübersicht  

Das neuronale Netz wird anhand des **MNIST‑Datensatzes** (handgeschriebene Ziffern 0–9) trainiert.  
Zusätzlich bietet das Projekt die Möglichkeit, eigene Ziffern zu zeichnen und manuell zu labeln,  
um Vorhersagen außerhalb des klassischen Trainingsdatensatzes zu testen.  

## Motivation & Lernziel  

Dieses Projekt dient primär Lernzwecken und verfolgt das Ziel, folgende Aspekte besser zu verstehen:  

- Aufbau und Funktionsweise neuronaler Netze  
- Trainingsprozess (Forward- und Backpropagation)  
- Einfluss von Netzwerkarchitektur auf die Genauigkeit  

  
## Bekannte Einschränkungen  

Das aktuelle Modell basiert auf einem **fully connected neuronalen network (FCNN)**.  

Während die Genauigkeit bei standardisierten MNIST‑Testdaten akzeptabel ist,  
sinkt die Erkennungsrate deutlich bei **frei gezeichneten Ziffern**.  

Dies liegt hauptsächlich daran, dass:  
- räumliche Bildmerkmale nicht explizit berücksichtigt werden  
- selbst gezeichnete Ziffern sich in Strichstärke, Positionierung und Rauschen  
  deutlich vom MNIST‑Datensatz unterscheiden  

Diese Einschränkung verdeutlicht die Grenzen einfache Netze bei Bilddaten.  


## Geplante Erweiterungen  

Als nächster Schritt ist die Implementierung eines **Convolutional Neural Networks (CNN)** geplant.  

Ein CNN eignet sich besser für Bilddaten, da:  
- räumliche Strukturen erkannt werden können  
- lokale Merkmale wie Kanten und Formen extrahiert werden  
- die Generalisierung auf frei gezeichnete Ziffern verbessert wird  

Ziel ist es, die Genauigkeit bei benutzerdefinierten Eingaben deutlich zu erhöhen  
und ein robusteres Modell zu erstellen.  

## Architektur  
- Input Layer : 784 Neuronen (28x28 Pixel)  
- Hidden Layer 1: 128 Neuronen, Aktivierung: Relu 
- Hidden Layer 2: 64 Neuronen, Aktivierung: Relu  
- Output Layer : 10 Neuronen, Aktivierung: Softmax  

## Ergebnisse nach 5 Epochen  
- Genauigkeit: 96,99%  
- Korrekte Vorhersagen: 9 699  
- Falsche Vorhersagen: 301  
- Durchschnittliche Sicherheit: 98,12%  
