import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.petri_net.visualizer import Variants
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.visualization.heuristics_net import visualizer as hn_vis


#Wczytywania event logu
df = pd.read_csv('../Data_source/Event_log.csv', sep=',')
print("Wczytywanie Event Logu zakonczone")

#Konwersja even logu na data frame zgodnie z wymogami PM4PY
df = pm4py.format_dataframe(df, case_id='Order_ID', activity_key='Activity', timestamp_key='Timestamp')
event_log = pm4py.convert_to_event_log(df)
print("Konwersja Data Frame na Event Log zakonczona")

#Odkrywanie Petri Net z Alpha Miner

net, im, fm = pm4py.discover_petri_net_alpha(event_log)
petri_net_visualization = pn_vis.apply(net, im, fm, log=event_log)
pn_vis.view(petri_net_visualization)
pn_vis.save(petri_net_visualization, "petri_net_alpha_miner.png")
print("Odkrylem Petri Net z Alpha Miner. Wizualizacja zostala wyswietlona i zapisana jako petri_net_alpha_miner.png ")

#Odkrywanie Petri Net z Inductive Miner
net, im, fm = pm4py.discover_petri_net_inductive(event_log)
petri_net_visualization = pn_vis.apply(net, im, fm, log=event_log)
pn_vis.view(petri_net_visualization)
pn_vis.save(petri_net_visualization,"petri_net_inductive_miner.png")
print("Odkrylem Petri Net z Inductive Miner. Wizualizacja zostala wyswietlona i zapisana jako petri_net_inductive_miner.png ")

#Zastosowanie parametrow

#FREQUENCY
net, im, fm = pm4py.discover_petri_net_inductive(event_log)
petri_net_visualization = pn_vis.apply(net, im, fm, log=event_log, variant=Variants.FREQUENCY)
pn_vis.view(petri_net_visualization)
pn_vis.save(petri_net_visualization, "petri_net_inductive_miner_frequency.png")
print("Odkrylem Petri Net z Inductive Miner i wariantem FREQUENCY. Wizualizacja zostala wyswietlona i zapisana jako petri_net_inductive_miner_frequency.png ")

#PERFORMANCE
net, im, fm = pm4py.discover_petri_net_inductive(event_log)
petri_net_visualization = pn_vis.apply(net, im, fm, log=event_log, variant=Variants.PERFORMANCE)
pn_vis.view(petri_net_visualization)
pn_vis.save(petri_net_visualization, "petri_net_inductive_miner_performance.png")
print("Odkrylem Petri Net z Inductive Miner i wariantem PERFORMANCE. Wizualizacja zostala wyswietlona i zapisana jako petri_net_inductive_miner_performance.png ")

#ALIGMENTS
#Wczytywanie drugiego logu (do sprawdzenia dopasowania)
df_new = pd.read_csv('../Data_source/log_aligments.csv', sep=',')
print("Wczytywanie Event Logu alignments zakonczone")
df_new = pm4py.format_dataframe(df_new, case_id='Order_ID', activity_key='Activity', timestamp_key='Timestamp')
event_log_aligments = pm4py.convert_to_event_log(df_new)
print("Konwersja Data Frame na Event Log zakonczona")
petri_net_visualization = pn_vis.apply(net, im, fm, log=event_log_aligments, variant=Variants.ALIGNMENTS)
pn_vis.view(petri_net_visualization)
pn_vis.save(petri_net_visualization, 'petri_net_inductive_miner_aligments.png')
print('Dopasowanie nowego event logu wzgledem poczatkowego zostalo wyswietlone i zapisane jako petri_net_inductive_miner_aligments.png')

#NOISE_THRESHOLD

noise_threshold_parameter = [0, 0.1, 0.2, 0.5, 1]

for i in noise_threshold_parameter:
   net, im, fm = pm4py.discover_petri_net_inductive(event_log, noise_threshold=i)
   petri_net_visualization_noise_threshold = pn_vis.apply(net, im, fm, log=event_log)
   pn_vis.view(petri_net_visualization_noise_threshold)
   pn_vis.save(petri_net_visualization_noise_threshold, f"petri_net_visualization_noise_threshold_{i}.png")
   print(f"Wizualizacja dla noise_threshold={i} została zapisana i wyświetlona.")
   filtered_log = pm4py.algo.filtering.log.variants.variants_filter.filter_variants_by_coverage_percentage(event_log,min_coverage_percentage=i)
   print(f"Ilosc wariantow dla {i} noise threshold:", len(pm4py.get_variants(filtered_log)))

#FILTROWANIE + NOISE_THRESHOLD

if len(pm4py.get_variants(filtered_log)) > 0:
   net, im, fm = pm4py.discover_petri_net_inductive(filtered_log, noise_threshold=i)
   petri_net_visualization_noise_threshold_filter = pn_vis.apply(net, im, fm, log=filtered_log)
   pn_vis.view(petri_net_visualization_noise_threshold_filter)
   pn_vis.save(petri_net_visualization_noise_threshold_filter, f"petri_net_visualization_noise_threshold_filter{i}.png")
   print(f"Wizualizacja dla noise_threshold={i} została zapisana i wyświetlona.")
else:
   print(f"Brak wariantów dla {i}, pomijam wizualizację.")

#HEURISTIC NET

heuristic_net = pm4py.discover_heuristics_net(event_log)
heuristic_net_visualization = hn_vis.apply(heuristic_net)
hn_vis.view(heuristic_net_visualization)
hn_vis.save(heuristic_net_visualization, "heuristic_net.png")

#HEURISTIC NET WITH PARAMETERS

heuristic_net_parameters_value = [0.1, 0.7, 0.95]

for i in heuristic_net_parameters_value:
    heuristic_net_parameters = pm4py.discover_heuristics_net(event_log, dependency_threshold= i, and_threshold= i, loop_two_threshold= i)
    heuristic_net_visualization_parameters = hn_vis.apply(heuristic_net_parameters)
    hn_vis.view(heuristic_net_visualization_parameters)
    hn_vis.save(heuristic_net_visualization_parameters, f"heuristic_net_parameters{i}.png")
    print(f"Wizualizacja dla heuristic_net_parameters_value={i} została wyswietlona i zapisana.")



