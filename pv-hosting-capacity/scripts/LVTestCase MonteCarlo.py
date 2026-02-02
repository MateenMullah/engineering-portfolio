# === EU LV Daily Time-Series PV Hosting Capacity Analysis - FIXED ===

import os, math, random, csv
import win32com.client as win32
import matplotlib.pyplot as plt
import numpy as np

# ---------------- USER SETTINGS ----------------
MASTER              = r"C:\OpenDSS_test\LVTestCaseUrban\Master.dss"
LOADS_FILE          = r"C:\OpenDSS_test\LVTestCaseUrban\Loads.txt"
OUTPUT_CSV          = r"C:\OpenDSS_test\LVTestCaseUrban\timeseries_results.csv"
VIOLATIONS_CSV      = r"C:\OpenDSS_test\LVTestCaseUrban\violations_detailed.csv"
LINE_CURRENTS_CSV   = r"C:\OpenDSS_test\LVTestCaseUrban\line_max_currents.csv"

START_HOUR          = 0
END_HOUR            = 24
TIMESTEP_MIN        = 15
LOAD_SCALE_FACTOR   = 4.7

N_PV_SITES          = 10
SEED                = 0
PV_SCALE_FACTORS    = [100]
PV_BASE_KW          = 5
ENABLE_VOLTVAR      = True

VMIN_LIM            = 0.90
VMAX_LIM            = 1.10
VUF_MAX             = 0.03

MONITOR_LOADS = ["LOAD1"]

def get_irradiance(hour):
    if hour < 6 or hour >= 20:
        return 0.0
    solar_hour = hour - 6
    peak_hour = 7
    if solar_hour <= peak_hour:
        return math.sin((solar_hour / peak_hour) * (math.pi / 2)) ** 2
    else:
        hours_to_sunset = 14 - solar_hour
        return math.sin((hours_to_sunset / 7) * (math.pi / 2)) ** 2

def dss():
    for prog in ("OpenDSSengine.DSS","OpenDSSEngine.DSS"):
        try: 
            dss_obj = win32.Dispatch(prog)
            # Suppress OpenDSS popup windows
            dss_obj.AllowForms = False
            return dss_obj
        except Exception: 
            pass
    raise RuntimeError("Could not create OpenDSS COM object")

def FIRST(coll):
    try:
        return coll.First()
    except TypeError:
        return coll.First

def NEXT(coll):
    try:
        return coll.Next()
    except TypeError:
        return coll.Next

def compile_and_setup(text, sol):
    text.Command = "clear"
    text.Command = f'compile "{MASTER}"'
    text.Command = "batchedit loadshape..* useactual=no"
    if LOAD_SCALE_FACTOR != 1.0:
        text.Command = f"batchedit load..* kW={LOAD_SCALE_FACTOR}"
    text.Command = "set mode=yearly"
    text.Command = f"set stepsize={TIMESTEP_MIN}m"
    total_steps = int((END_HOUR - START_HOUR) * 60 / TIMESTEP_MIN)
    text.Command = f"set number={total_steps}"
    text.Command = "set controlmode=time"
    text.Command = "set maxcontroliter=100"
    text.Command = "CalcVoltageBases"
    # Define Urban 
    #text.Command = "new XYCurve.vv_urban_A npts=4 xarray=[0.88,0.97,1.03,1.10] yarray=[0.60,0.00,0.00,-0.44]"
    #Define Suburban
    text.Command = "new XYCurve.vv_suburban_B npts=4 xarray=[0.88,0.97,1.03,1.10] yarray=[0.44,0.00,0.00,-0.44]"
    return total_steps

def parse_loads_txt(filepath):
    """Parse Loads.txt: {load_name: (bus_name, phase)}"""
    load_info = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                if 'Load.' in line:
                    parts = line.split()
                    load_name = None
                    bus_info = None
                    for part in parts:
                        if 'Load.' in part:
                            load_name = part.split('.')[1].upper()
                        elif 'Bus1=' in part or 'bus1=' in part:
                            bus_info = part.split('=')[1]
                    if load_name and bus_info:
                        if '.' in bus_info:
                            bus_name, phase_str = bus_info.split('.')
                            phase = int(phase_str)
                        else:
                            bus_name = bus_info
                            phase = 1
                        load_info[load_name] = (bus_name, phase)
        return load_info
    except FileNotFoundError:
        return {}

def get_existing_loads_from_circuit(circuit):
    """Get set of load names that actually exist in compiled circuit"""
    existing = set()
    loads = circuit.Loads
    i = FIRST(loads)
    while i > 0:
        existing.add(loads.Name.upper())
        i = NEXT(loads)
    return existing

def get_all_pv_names(circuit):
    circuit.SetActiveClass("PVSystem")
    try:
        return list(circuit.ActiveClass.AllNames)
    except Exception:
        return []

def worst_voltage_report(circuit):
    vmin, vmin_bus, vmin_ph =  9.0, "", 0
    vmax, vmax_bus, vmax_ph = 0.0, "", 0
    for b in list(circuit.AllBusNames):
        circuit.SetActiveBus(b)
        if circuit.ActiveBus.kVBase <= 0: 
            continue
        nodes = list(circuit.ActiveBus.Nodes)
        puVA  = list(circuit.ActiveBus.puVmagAngle)
        for i, n in enumerate(nodes):
            if n not in (1, 2, 3): 
                continue
            if 2*i >= len(puVA):
                break
            vpu = puVA[2*i]
            if vpu < vmin and vpu > 0.001:
                vmin, vmin_bus, vmin_ph = vpu, b, n
            if vpu > vmax:
                vmax, vmax_bus, vmax_ph = vpu, b, n
    return vmin, vmin_bus, vmin_ph, vmax, vmax_bus, vmax_ph

def check_voltage_unbalance(circuit, hour_float, step):
    max_vuf = 0.0
    worst_bus = ""
    worst_bus_details = None
    for bus_name in list(circuit.AllBusNames):
        circuit.SetActiveBus(bus_name)
        num_nodes = circuit.ActiveBus.NumNodes
        if num_nodes == 3:
            voltages = list(circuit.ActiveBus.puVmagAngle)
            v_mags = [voltages[i] for i in range(0, len(voltages), 2)]
            v_angs = [voltages[i] for i in range(1, len(voltages), 2)]
            if len(v_mags) == 3 and len(v_angs) == 3:
                v1 = v_mags[0] * np.exp(1j * np.radians(v_angs[0]))
                v2 = v_mags[1] * np.exp(1j * np.radians(v_angs[1]))
                v3 = v_mags[2] * np.exp(1j * np.radians(v_angs[2]))
                a = np.exp(1j * 2 * np.pi / 3)
                v_pos = (v1 + a * v2 + a**2 * v3) / 3
                v_neg = (v1 + a**2 * v2 + a * v3) / 3
                vuf = abs(v_neg) / abs(v_pos) if abs(v_pos) > 0 else 0
                if vuf > max_vuf:
                    max_vuf = vuf
                    worst_bus = bus_name
                    worst_bus_details = {
                        'bus': bus_name,
                        'vuf': vuf,
                        'v_phase_a_mag': v_mags[0],
                        'v_phase_b_mag': v_mags[1],
                        'v_phase_c_mag': v_mags[2],
                        'v_phase_a_ang': v_angs[0],
                        'v_phase_b_ang': v_angs[1],
                        'v_phase_c_ang': v_angs[2],
                        'v_spread': max(v_mags) - min(v_mags),
                        'v_pos_mag': abs(v_pos),
                        'v_neg_mag': abs(v_neg),
                        'timestep': step,
                        'hour': hour_float
                    }
    return max_vuf, worst_bus, worst_bus_details

def check_all_thermal_violations(circuit):
    violations = []
    lines = circuit.Lines
    i = FIRST(lines)
    while i > 0:
        name = lines.Name
        linecode = lines.LineCode
        try:
            norm = float(lines.NormAmps)
        except:
            norm = 0.0
        if norm > 0:
            circuit.SetActiveElement(f"Line.{name}")
            ce = circuit.ActiveCktElement
            if ce:
                mags = list(ce.CurrentsMagAng)[0::2]
                num_phases = ce.NumPhases
                from_end_mags = mags[:num_phases]
                imax = max(from_end_mags) if from_end_mags else 0.0
                if imax > norm:
                    violations.append({
                        'line_name': name,
                        'linecode': linecode,
                        'current_A': imax,
                        'rating_A': norm,
                        'overload_pct': (imax / norm * 100.0)
                    })
        i = NEXT(lines)
    return violations

def worst_line_loading(circuit):
    worst = (0.0, "", 0.0, 0.0)
    lines = circuit.Lines
    i = FIRST(lines)
    while i > 0:
        name = lines.Name
        try: norm = float(lines.NormAmps)
        except: norm = 0.0
        circuit.SetActiveElement(f"Line.{name}")
        ce = circuit.ActiveCktElement
        if ce:
            mags = list(ce.CurrentsMagAng)[0::2]
            num_phases = ce.NumPhases
            from_end_mags = mags[:num_phases]
            imax = max(from_end_mags) if from_end_mags else 0.0
        else:
            imax = 0.0
        pct  = (imax/norm*100.0) if norm>0 else 0.0
        if pct > worst[0]: 
            worst = (pct, name, imax, norm)
        i = NEXT(lines)
    return worst

def get_all_line_currents(circuit):
    line_data = {}
    lines = circuit.Lines
    i = FIRST(lines)
    while i > 0:
        name = lines.Name
        circuit.SetActiveElement(f"Line.{name}")
        ce = circuit.ActiveCktElement
        if ce:
            all_currents = list(ce.CurrentsMagAng)
            mags = all_currents[0::2]
            num_phases = ce.NumPhases
            from_end_mags = mags[:num_phases]
            imax = max(from_end_mags) if from_end_mags else 0.0
            line_data[name] = imax
        i = NEXT(lines)
    return line_data

def get_substation_loading(circuit):
    try:
        xfmrs = circuit.Transformers
        i = FIRST(xfmrs)
        if i > 0:
            name = xfmrs.Name
            circuit.SetActiveElement(f"Transformer.{name}")
            ce = circuit.ActiveCktElement
            if ce:
                powers = list(ce.Powers)
                ncond = ce.NumConductors
                P1 = sum(powers[0:2*ncond:2])
                Q1 = sum(powers[1:2*ncond:2])
                S1 = math.sqrt(P1**2 + Q1**2)
                return S1, P1, Q1
    except Exception:
        pass
    return 0.0, 0.0, 0.0

def get_total_load_power(circuit):
    total_p = 0.0
    total_q = 0.0
    loads = circuit.Loads
    i = FIRST(loads)
    while i > 0:
        load_name = loads.Name
        circuit.SetActiveElement(f"Load.{load_name}")
        ce = circuit.ActiveCktElement
        if ce:
            powers = list(ce.Powers)
            for j in range(0, len(powers), 2):
                total_p += abs(powers[j])
            for j in range(1, len(powers), 2):
                total_q += abs(powers[j])
        i = NEXT(loads)
    return total_p, total_q

# === CHANGED: use PV names instead of enumerator so all PVs are included ===
def get_total_pv_power(circuit):
    P = Q = 0.0
    pv_names = get_all_pv_names(circuit)
    pvs = circuit.PVsystems
    for name in pv_names:
        try:
            pvs.Name = name
            P += pvs.kW
            Q += pvs.kvar
        except Exception:
            pass
    return P, Q

if __name__ == "__main__":
    print("\n" + "="*90)
    print("EU LOW-VOLTAGE DISTRIBUTION NETWORK ANALYSIS")
    print("="*90)
    
    dss_eng = dss()
    text    = dss_eng.Text
    circuit = dss_eng.ActiveCircuit
    sol     = circuit.Solution
    
    all_timestep_data = []
    all_violations = []
    line_max_currents = {}
    
    for scale_idx, pv_scale in enumerate(PV_SCALE_FACTORS):
        print(f"SCENARIO {scale_idx+1}: PV SCALE = {pv_scale}% (Base PV={PV_BASE_KW} kW)")
        
        total_steps = compile_and_setup(text, sol)
        
        if N_PV_SITES > 0:
            random.seed(SEED)
            
            all_loads_from_file = parse_loads_txt(LOADS_FILE)
            existing_loads = get_existing_loads_from_circuit(circuit)
            valid_loads = all_loads_from_file
            
            print(f"PV SITE SELECTION:")
            print(f"  Loads in file: {len(all_loads_from_file)}")
            print(f"  Loads in circuit: {len(existing_loads)}")
            print(f"  Valid loads for PV: {len(valid_loads)}")
            print(f"  Requested PV sites: {N_PV_SITES}")
            print(f"  Random seed: {SEED}\n")
            
            if len(valid_loads) == 0:
                print("No valid loads found!")
                continue
            
            # Select loads for PV
            valid_load_names = list(valid_loads.keys())
            num_pv_to_add = min(N_PV_SITES, len(valid_load_names))
            selected_loads = random.sample(valid_load_names, num_pv_to_add)
            
            pv_kW_scaled = PV_BASE_KW * (pv_scale / 100.0)
            
            print(f"Adding {len(selected_loads)} PV systems at {pv_kW_scaled:.2f} kW each...")
            
            phase_counts = {1: 0, 2: 0, 3: 0}
            
            for idx, load_name in enumerate(selected_loads):
                bus_name, phase = valid_loads[load_name]
                phase_counts[phase] += 1
                
                pv_name = f"PV{idx+1:02d}"
                
                cmd = (
                    f'New PVSystem.{pv_name} bus1={bus_name}.{phase} phases=1 '
                    f'kV=0.23 kVA={pv_kW_scaled} Pmpp={pv_kW_scaled} '
                    f'irradiance=1.0 enabled=yes %cutin=0.1 %cutout=0.1'
                )
                text.Command = cmd

                # Echo any OpenDSS parser/creation message for this PV
                res = str(text.Result).strip()
                if res:
                    print(f"    -> OpenDSS: {res}")
                
                if ENABLE_VOLTVAR:
                    text.Command = "new XYCurve.vv_rural_aggressive npts=4 xarray=[0.88,0.97,1.03,1.10] yarray=[0.60,0.00,0.00,-0.60]"
                    
                
                if idx < 5:
                    print(f"  {pv_name}: Load={load_name}, Bus={bus_name}.{phase}")
            
            if len(selected_loads) > 5:
                print(f"  ... and {len(selected_loads) - 5} more")
            
            print(f"\nPhase distribution:")
            print(f"  Phase A (1): {phase_counts[1]} PV systems")
            print(f"  Phase B (2): {phase_counts[2]} PV systems")
            print(f"  Phase C (3): {phase_counts[3]} PV systems\n")
            
            text.Command = "CalcVoltageBases"
            sol.Solve()
            
            # Verify PV count using robust ActiveClass names
            pv_names = get_all_pv_names(circuit)
            pv_count = len(pv_names)
            
            print(f"{'='*70}")
            print(f"PV VERIFICATION:")
            print(f"  Attempted: {len(selected_loads)} | Added: {pv_count}")
            print(f"{'='*70}")
            
            if pv_count == len(selected_loads):
                print(f"✓ All {pv_count} PV systems successfully added!\n")
            else:
                print(f"✗ WARNING: Only {pv_count}/{len(selected_loads)} PV added!\n")
        
        # This total now includes all PVs thanks to name-based iteration
        P_pv_test, Q_pv_test = get_total_pv_power(circuit)
        print(f"Initial PV: P={P_pv_test:.2f} kW, Q={Q_pv_test:.2f} kvar")
        print(f"(Zero expected - irradiance starts at 0.0)\n")
        
        print(f"Running time-series: {START_HOUR} to {END_HOUR} hrs, step={TIMESTEP_MIN} min")
        print(f"Total timesteps: {total_steps}")
        
        violations_this_scale = []
        min_vmin, max_vmax = 9.0, 0.0
        max_line_pct = 0.0
        max_vuf_seen = 0.0
        load_I_logs = {lname: [] for lname in MONITOR_LOADS}
        
        for step in range(total_steps):
            hour_float = START_HOUR + (step * TIMESTEP_MIN / 60.0)
            hour_int = int(hour_float)
            minute = int((hour_float - hour_int) * 60)
            text.Command = f"set hour={hour_int}"
            text.Command = f"set sec={minute * 60}"
            
            # === CHANGED: update irradiance for ALL PVs by name ===
            irr_val = get_irradiance(hour_float)
            if N_PV_SITES > 0:
                pvs = circuit.PVsystems
                for name in get_all_pv_names(circuit):
                    try:
                        pvs.Name = name
                        pvs.irradiance = irr_val
                    except Exception:
                        pass
            
            sol.Solve()
            
            vmin, bmin, phmin, vmax, bmax, phmax = worst_voltage_report(circuit)
            min_vmin = min(min_vmin, vmin)
            max_vmax = max(max_vmax, vmax)
            line_pct, line_name, line_I, line_norm = worst_line_loading(circuit)
            max_line_pct = max(max_line_pct, line_pct)
            thermal_viols = check_all_thermal_violations(circuit)
            vuf, vuf_bus, vuf_details = check_voltage_unbalance(circuit, hour_float, step)
            max_vuf_seen = max(max_vuf_seen, vuf)
            current_line_currents = get_all_line_currents(circuit)
            for lname, curr in current_line_currents.items():
                if lname not in line_max_currents:
                    line_max_currents[lname] = curr
                else:
                    line_max_currents[lname] = max(line_max_currents[lname], curr)
            P_load, Q_load = get_total_load_power(circuit)
            P_pv, Q_pv = get_total_pv_power(circuit)
            S_sub, P_sub, Q_sub = get_substation_loading(circuit)
            
            loads = circuit.Loads
            i = FIRST(loads)
            while i > 0:
                lname_upper = loads.Name.upper()
                if lname_upper in MONITOR_LOADS:
                    circuit.SetActiveElement(f"Load.{loads.Name}")
                    ce = circuit.ActiveCktElement
                    if ce:
                        Imags = list(ce.CurrentsMagAng)[0::2]
                        num_phases = ce.NumPhases
                        from_end_I = Imags[:num_phases]
                        I1 = from_end_I[0] if len(from_end_I) > 0 else 0.0
                        I2 = from_end_I[1] if len(from_end_I) > 1 else 0.0
                        I3 = from_end_I[2] if len(from_end_I) > 2 else 0.0
                        Imax = max(from_end_I) if from_end_I else 0.0
                        load_I_logs[lname_upper].append({
                            'pv_scale': pv_scale,
                            'timestep': step,
                            'hour': hour_float,
                            'I_A_phase1': I1,
                            'I_A_phase2': I2,
                            'I_A_phase3': I3,
                            'I_A_max': Imax
                        })
                i = NEXT(loads)
            
            viols = []
            if vmin < VMIN_LIM:
                viols.append({
                    'scenario': f'PV_{pv_scale}%',
                    'timestep': step,
                    'hour': hour_float,
                    'type': 'undervoltage',
                    'location': f'{bmin}.{phmin}',
                    'value': vmin,
                    'limit': VMIN_LIM,
                    'severity': f'{(VMIN_LIM-vmin)*100:.2f}%'
                })
            if vmax > VMAX_LIM:
                viols.append({
                    'scenario': f'PV_{pv_scale}%',
                    'timestep': step,
                    'hour': hour_float,
                    'type': 'overvoltage',
                    'location': f'{bmax}.{phmax}',
                    'value': vmax,
                    'limit': VMAX_LIM,
                    'severity': f'{(vmax-VMAX_LIM)*100:.2f}%'
                })
            for thermal_viol in thermal_viols:
                viols.append({
                    'scenario': f'PV_{pv_scale}%',
                    'timestep': step,
                    'hour': hour_float,
                    'type': 'thermal',
                    'location': f'{thermal_viol["line_name"]} (LineCode: {thermal_viol["linecode"]})',
                    'value': thermal_viol['current_A'],
                    'limit': thermal_viol['rating_A'],
                    'severity': f'{(thermal_viol["overload_pct"] - 100):.1f}%'
                })
            if vuf > VUF_MAX:
                viols.append({
                    'scenario': f'PV_{pv_scale}%',
                    'timestep': step,
                    'hour': hour_float,
                    'type': 'voltage_unbalance',
                    'location': vuf_bus,
                    'value': vuf,
                    'limit': VUF_MAX,
                    'severity': f'{(vuf - VUF_MAX)*100:.2f}%'
                })
            violations_this_scale.extend(viols)
            all_violations.extend(viols)
            all_timestep_data.append({
                'scenario': f'PV_{pv_scale}%',
                'timestep': step,
                'hour': hour_float,
                'irradiance': irr_val,
                'pv_p_kw': P_pv,
                'pv_q_kvar': Q_pv,
                'load_p_kw': P_load,
                'load_q_kvar': Q_load,
                'vmin': vmin,
                'vmin_bus': f"{bmin}.{phmin}",
                'vmax': vmax,
                'vmax_bus': f"{bmax}.{phmax}",
                'max_vuf': vuf,
                'vuf_bus': vuf_bus,
                'max_line_pct': line_pct,
                'num_thermal_violations': len(thermal_viols),
                'sub_kva': S_sub,
                'sub_p_kw': P_sub,
                'sub_q_kvar': Q_sub,
                'has_violation': len(viols) > 0
            })
            
            if (step) % 24 == 0:
                print(f"  Step {step:4d} | Hour {hour_float:2.0f} | "
                      f"Vmin={vmin:.4f} Vmax={vmax:.4f} | "
                      f"VUF={vuf*100:.2f}% | "
                      f"Line={line_pct:.1f}% | "
                      f"P_load={P_load:.1f} P_pv={P_pv:.1f} kW")

        _outdir = os.path.dirname(MASTER)
        for lname, rows in load_I_logs.items():
            if rows:
                outpath = os.path.join(_outdir, f"load_current_timeseries_{lname}_S{scale_idx+1}.csv")
                with open(outpath, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader(); w.writerows(rows)
        
        print(f"\n{'='*90}")
        print(f"SCENARIO {scale_idx+1} SUMMARY (PV Scale={pv_scale}%)")
        print(f"{'='*90}")
        print(f"Min voltage: {min_vmin:.4f} pu ({'OK' if min_vmin >= VMIN_LIM else 'VIOLATION'})")
        print(f"Max voltage: {max_vmax:.4f} pu ({'OK' if max_vmax <= VMAX_LIM else 'VIOLATION'})")
        print(f"Max VUF: {max_vuf_seen*100:.2f}% ({'OK' if max_vuf_seen <= VUF_MAX else 'VIOLATION'})")
        print(f"Max line loading: {max_line_pct:.2f}%")
        print(f"Violations: {len(violations_this_scale)} timesteps\n")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        if all_timestep_data:
            writer = csv.DictWriter(f, fieldnames=all_timestep_data[0].keys())
            writer.writeheader()
            writer.writerows(all_timestep_data)
    print(f"✓ Data: {OUTPUT_CSV}")

    if all_violations:
        with open(VIOLATIONS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_violations[0].keys())
            writer.writeheader()
            writer.writerows(all_violations)
    
    with open(LINE_CURRENTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Line_Name', 'Max_Current_A'])
        for line_name in sorted(line_max_currents.keys()):
            max_i = line_max_currents[line_name]
            writer.writerow([line_name, f"{max_i:.2f}"])
    print(f"✓ Line currents: {LINE_CURRENTS_CSV}")

    print(f"\n{'='*90}")
    print("GENERATING GRAPHS")
    print(f"{'='*90}")

    OUTPUT_DIR = os.path.dirname(MASTER)

    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        ts_data = list(reader)

    hours = [float(row['hour']) for row in ts_data]
    load_p_kw = [float(row['load_p_kw']) for row in ts_data]
    load_q_kvar = [float(row['load_q_kvar']) for row in ts_data]
    pv_p_kw = [float(row['pv_p_kw']) for row in ts_data]
    pv_q_kvar = [float(row['pv_q_kvar']) for row in ts_data]
    sub_p_kw = [float(row['sub_p_kw']) for row in ts_data]
    sub_q_kvar = [float(row['sub_q_kvar']) for row in ts_data]
    vmin_vals = [float(row['vmin']) for row in ts_data]
    vmax_vals = [float(row['vmax']) for row in ts_data]
    vuf_vals = [float(row['max_vuf']) for row in ts_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Load and PV Power Profile (24 hours)', fontsize=16, fontweight='bold')
    ax1.plot(hours, load_p_kw, 'b-', linewidth=2, label='Load P (kW)')
    ax1.plot(hours, pv_p_kw, 'orange', linewidth=2, label='PV P (kW)')
    ax1.fill_between(hours, pv_p_kw, alpha=0.3, color='orange')
    ax1.set_ylabel('Active Power (kW)', fontsize=12, fontweight='bold')
    ax1.set_title('Active Power: Load vs PV', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(START_HOUR, END_HOUR)
    ax2.plot(hours, load_q_kvar, 'b-', linewidth=2, label='Load Q (kvar)')
    ax2.plot(hours, pv_q_kvar, 'r-', linewidth=2, label='PV Q (kvar)')
    ax2.set_ylabel('Reactive Power (kvar)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_title('Reactive Power', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_xlim(START_HOUR, END_HOUR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'load_pv_power_profile.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Load/PV graph")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hours, vmin_vals, 'b-', linewidth=2, label='Min Voltage')
    ax.plot(hours, vmax_vals, 'r-', linewidth=2, label='Max Voltage')
    ax.axhline(y=VMIN_LIM, color='b', linestyle='--', alpha=0.5, label=f'Min Limit ({VMIN_LIM} pu)')
    ax.axhline(y=VMAX_LIM, color='r', linestyle='--', alpha=0.5, label=f'Max Limit ({VMAX_LIM} pu)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (pu)', fontsize=12, fontweight='bold')
    ax.set_title('Voltage Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(START_HOUR, END_HOUR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'voltage_profile_timeseries.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Voltage graph")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    vuf_pct = [v * 100 for v in vuf_vals]
    ax.plot(hours, vuf_pct, 'purple', linewidth=2, label='VUF')
    ax.axhline(y=VUF_MAX * 100, color='r', linestyle='--', alpha=0.5, label=f'Limit ({VUF_MAX*100:.1f}%)')
    ax.fill_between(hours, vuf_pct, alpha=0.3, color='purple')
    ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax.set_ylabel('VUF (%)', fontsize=12, fontweight='bold')
    ax.set_title('Voltage Unbalance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim(START_HOUR, END_HOUR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'voltage_unbalance_timeseries.png'), dpi=300, bbox_inches='tight')
    print(f"✓ VUF graph")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hours, sub_p_kw, 'b-', linewidth=2, label='P')
    ax.plot(hours, sub_q_kvar, 'r-', linewidth=2, label='Q')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (kW/kvar)', fontsize=12, fontweight='bold')
    ax.set_title('Substation Power', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(START_HOUR, END_HOUR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'substation_power_timeseries.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Substation graph")
    plt.close()
    
    print(f"\n{'='*90}")
    print("POWER SUMMARY")
    print(f"{'='*90}")
    print(f"\nLoad: P={min(load_p_kw):.1f}-{max(load_p_kw):.1f} kW (avg {sum(load_p_kw)/len(load_p_kw):.1f})")
    print(f"PV: P={min(pv_p_kw):.1f}-{max(pv_p_kw):.1f} kW (avg {sum(pv_p_kw)/len(pv_p_kw):.1f})")
    print(f"Substation: P={min(sub_p_kw):.1f}-{max(sub_p_kw):.1f} kW (avg {sum(sub_p_kw)/len(sub_p_kw):.1f})")
    if min(sub_p_kw) < 0:
        print(f"  (Negative = reverse power flow from PV)")
    print(f"\nVoltage: {min(vmin_vals):.4f}-{max(vmax_vals):.4f} pu")
    print(f"Max VUF: {max(vuf_vals)*100:.2f}% ({'✓ OK' if max(vuf_vals) <= VUF_MAX else '✗ VIOLATION'})")
    print(f"Max line loading: {max(line_max_currents.values()) if line_max_currents else 0:.2f} A")
    print(f"\n{'='*90}")
    print("COMPLETE!")
    print(f"{'='*90}\n")