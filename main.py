from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


def main():
    information = """
    Jan Władysław Piltz (ur. 15 stycznia 1870 w Aleksandrowie, zm. 26 listopada 1930 w Krakowie) – polski lekarz neurolog i psychiatra, profesor Uniwersytetu Jagiellońskiego i pierwszy kierownik Katedry Neurologiczno-Psychiatrycznej Uniwersytetu Jagiellońskiego, twórca krakowskiej szkoły neurologicznej, działacz społeczny.

Życiorys
Urodził się 15 stycznia 1870 roku w Aleksandrowie w powiecie nieszawskim, na terenie Guberni Warszawskiej, w spolonizowanej rodzinie o niemieckich korzeniach. Jego rodzicami byli Jan Piltz i Emma Schmidt. Ojciec Piltza był zawiadowcą stacji kolejowej, za udział w powstaniu styczniowym zesłanym na kilka lat na Syberię. Przyrodnim bratem Jana był polityk i dyplomata Erazm Piltz.

Uczęszczał do gimnazjum realnego w Warszawie, ukończył je w 1888 roku. W latach 1888–1889 studiował nauki przyrodnicze i matematykę na wydziale filozoficznym Uniwersytetu w Zurychu. Jego przyjacielem z tego okresu był Leon Marchlewski[1]. Po zdaniu matury w Bernie powrócił na Uniwersytet Zuryski w 1892 roku i podjął studia medyczne; ukończył je w 1895[2].

Jako student był asystentem w klinice anatomii i histologii Philippa Stöhra (od maja do listopada 1893) i, w tym samym czasie, w Instytucie Badań Mózgu u Constantina von Monakowa. Pracował też jako asystent w klinice psychiatrycznej Augusta Forela (od lutego do maja 1895), u Marc-André Oliveta i Johannèsa Martina w Genewie (od stycznia 1895 do marca 1896). W 1897 roku wyjechał do St. Petersburga, tam zdał uzupełniające egzaminy maturalne z łaciny i greki i nostryfikował dyplom lekarski na Uniwersytecie w Kazaniu. Przez rok pracował w klinice neurologii i psychiatrii Akademii Medyko-Chirurgicznej w St. Petersburgu i doktoryzował się pod kierunkiem Władimira Biechtieriewa. Na zaproszenie nowego kierownika kliniki w Zurychu, Eugena Bleulera, wrócił do Szwajcarii i został we wrześniu 1898 pierwszym asystentem kliniki. Od kwietnia 1899 do stycznia 1900 był wicedyrektorem kliniki psychiatrycznej Uniwersytetu w Lozannie, kierowanej przez Alberta Mahaima. Od stycznia do grudnia 1900 roku pracował w klinice neurologicznej Salpêtrière, kierowanej wówczas przez Jules′a Déjerine′a.

W 1901 znalazł się w Warszawie, zorganizował samodzielny oddział neurologiczny z pracownią neuropatologiczną w warszawskim Szpitalu Praskim. Doktoryzował się w Lozannie w 1904 roku u Forela na podstawie dysertacji Contribution à l′étude de la dissociation de la sensibilité douloureuse et thermique dans les cas de traumatisme et d′affection de la moëlle épinière.

23 kwietnia 1905 został powołany na stanowisko profesora nadzwyczajnego Uniwersytetu Jagiellońskiego i pierwszym kierownikiem Katedry Neurologiczno-Psychiatrycznej UJ założonej w tym samym roku. Piltz w ciągu dwudziestu lat rozbudował Klinikę na wzór kliniki psychiatrycznej w Monachium, tak że posiadała trzy duże pawilony i pracownie: anatomiczną, biochemiczną, histopatologiczną, metaboliczną i neurofizjologiczną. Jego asystentami i współpracownikami byli m.in. Eugeniusz Artwiński, Eugeniusz Brzezicki, Władysław Chłopicki, Jan Gallus, Włodzimierz Godłowski, Eufemiusz Herman, Hermann Nunberg, Adam Kunicki, Witold Łuniewski, Władysław Medyński, Aleksandra Mitrinowicz, Cezar Onufrowicz, Maksymilian Rose, Adam Rydel, Aurelia Sikorska, Karol Stanisław Szymański, Aleksander Ślączka, Marcin Zieliński[1].

Członek korespondent Royal College of Psychiatrists od 1930 roku[3]. Prezes Krakowskiego Towarzystwa Lekarskiego (1911), przewodniczący II Zjazdu Neurologów, Psychiatrów i Psychologów polskich (1912), przewodniczący Krakowskiego Towarzystwa Neurologiczno-Psychiatrycznego, prezes Krakowskiego Towarzystwa Eugenicznego, delegat Polski na Międzynarodowy Kongres Higieny Psychicznej w Waszyngtonie (1930)[1].

Był odznaczony Krzyżem Komandorskim Orderu św. Sawy.

Żonaty z Zofią Pawliczyńską (1876–1948). Mieli córkę Janinę (1902–1965), zamężną za Feliksem Siedleckim, i syna Jerzego (1903–1938), prawnika[1].

Pochowany jest w grobie rodzinnym na cmentarzu Rakowickim (kwatera 16)[4]. W Krakowie jego imieniem nazwano ulicę.
"""
    summary_template = """
    Given the information {information} about a person I want you to create:
1. A short summary
2. Two interesting facts about them
"""
    summary_prompt_template = PromptTemplate(
        input_variables={"information"},
        template=summary_template
    )
    llm = ChatOllama(temperature=0, model="gpt-oss:20b-cloud")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()