import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import pandas as pd

# Define brand colors
COLORS = {
    "gross_lactate": "#E6754E",
    "net_lactate": "#3E1111",
    "available_pyruvate": "#7E5B6A",
    "lactate_combustion": "#43444A",
    "at": "#191418",
    "lt1": "#A0A0A0",
    "fatmax": "#C0C0C0",
    "threshold": "#FFFFFF"
}

# Set font style
plt.rcParams["font.family"] = "Montserrat"

# Function to calculate VolRel dynamically
def calculate_volrel(body_mass, vo2max_power):
    return body_mass / (vo2max_power + 1e-9) if vo2max_power != 0 else 0

# Function to calculate VLaMax based on study formula
def calculate_vlamax(vo2max, bmi, body_mass, vo2max_power, sprint_power, age):
    if vo2max_power == 0:
        return 0
    mader_formula = 0.02049 / (vo2max_power / body_mass) * vo2max * (bmi / 22) * (1 + 0.000025 * age - 0.0000001 * body_mass)
    sprint_contribution = 0.000004 / (vo2max_power / body_mass) * sprint_power * (1 + 0.0000001 * age - 0.0000001 * body_mass)
    return mader_formula + sprint_contribution

# Function to calculate workload dynamically from test results
def calculate_workload(workload_3min, workload_6min, workload):
    if workload_3min > 0 and workload_6min > 0:
        return (workload_3min + workload_6min) / 3
    return workload

# Function to estimate fitness level
def estimate_fitness_level(vo2max, vlamax):
    if vo2max >= 65 and vlamax <= 0.5:
        return 'elite'
    elif vo2max >= 50 and vlamax <= 0.7:
        return 'advanced'
    elif vo2max >= 40 and vlamax <= 0.9:
        return 'intermediate'
    else:
        return 'beginner'

# Function to get glycogen storage per kg muscle based on fitness level
def get_glycogen_per_kg_muscle(fitness_level):
    levels = {
        'elite': 17,
        'advanced': 15,
        'intermediate': 14,
        'beginner': 13
    }
    return levels.get(fitness_level, 13)

# Function to estimate glycogen storage
def estimate_glycogen_storage(body_mass, body_fat_percentage, fitness_level, liver_glycogen=90, muscle_mass_percentage=0.70):
    fat_mass = body_mass * body_fat_percentage / 100
    lean_mass = body_mass - fat_mass
    muscle_mass = lean_mass * muscle_mass_percentage
    
    glycogen_per_kg_muscle = get_glycogen_per_kg_muscle(fitness_level)
    
    liver_glycogen_storage = liver_glycogen
    muscle_glycogen_storage = muscle_mass * glycogen_per_kg_muscle
    
    return liver_glycogen_storage + muscle_glycogen_storage

# Function to classify metabolic efficiency based on threshold values
def classify_efficiency(vlamax, at_percent, lt1_percent, fatmax_percent):
    # Define classification criteria based on thresholds and VLaMax
    if vlamax <= 0.5:
        if at_percent >= 85:
            return "Elite Endurance"
        elif at_percent >= 75:
            return "Advanced Endurance"
        else:
            return "Moderate Endurance"
    elif vlamax <= 0.7:
        if at_percent >= 80:
            return "Good Endurance"
        elif at_percent >= 70:
            return "Moderate Endurance"
        else:
            return "Basic Endurance"
    elif vlamax <= 0.9:
        if at_percent >= 75:
            return "Mixed Metabolism"
        elif at_percent >= 65:
            return "Basic Endurance"
        else:
            return "Limited Endurance"
    else:
        if fatmax_percent <= 50:
            return "Anaerobic Power"
        else:
            return "Power Oriented"

class MetabolicAnalyzer:
    def __init__(self, vo2max, body_mass, workload_3min, workload_6min, vlamax, vol_rel, sprint_power, vo2max_power, bmi, age, lactate_combustion_factor, Ks1=0.0631, Ks2=1.331):
        self.vo2max = max(vo2max, 1e-9)
        self.body_mass = body_mass
        self.sprint_power = sprint_power
        self.bmi = bmi
        self.age = age
        self.Ks1 = Ks1
        self.Ks2 = Ks2
        self.lactate_combustion_factor = lactate_combustion_factor
        
        self.vo2max_power = calculate_workload(workload_3min, workload_6min, vo2max_power)
        self.VolRel = calculate_volrel(body_mass, self.vo2max_power) if vol_rel == 0 else vol_rel
        self.vlamax = vlamax if vlamax != 0 else calculate_vlamax(self.vo2max, self.bmi, body_mass, self.vo2max_power, sprint_power, self.age)
        
        # Calculate relative power markers
        self.watt_O2 = self.vo2max_power / self.vo2max if self.vo2max > 0 else 0
        
        print(f"Calculated Values:\nVLaMax: {self.vlamax:.5f}\nVolRel: {self.VolRel:.5f}\nWorkload: {self.vo2max_power:.2f}")
    
    def generate_vo2_steady_state(self):
        return np.linspace(0, self.vo2max, 100)
    
    def calculate_lactate_dynamics(self):
        vo2ss = self.generate_vo2_steady_state()
        denom = np.maximum(self.vo2max - vo2ss, 1e-9)
        adp = np.sqrt((self.Ks1 * vo2ss) / denom)
        vlass = 60 * self.vlamax / (1 + (self.Ks2 / (adp ** 3)))
        la_comb = (self.lactate_combustion_factor / self.VolRel) * vo2ss
        vlanet = vlass - la_comb
        available_pyruvate = la_comb - vlass
        return vo2ss, vlass, vlanet, available_pyruvate, la_comb
    
    def calculate_intensity(self):
        vo2ss, vlass, vlanet, available_pyruvate, la_comb = self.calculate_lactate_dynamics()
        # Calculate overall demand
        overall_demand = vlass * (self.VolRel * self.body_mass) * ((1 / 4.3) * 22.4) / self.body_mass + vo2ss
        # Calculate intensity as percentage of VO2max
        intensity = (vo2ss / self.vo2max) * 100
        # Calculate power in watts
        power = vo2ss * self.watt_O2
        
        return intensity, power, overall_demand
    
    def find_thresholds(self):
        vo2ss, vlass, vlanet, available_pyruvate, la_comb = self.calculate_lactate_dynamics()
        intensity, power, overall_demand = self.calculate_intensity()
        
        # Find AT index (crossing point of lactate formation and combustion)
        at_indices = np.where(np.diff(np.sign(vlass - la_comb)) > 0)[0]
        at_index = at_indices[0] if len(at_indices) > 0 else len(vo2ss) // 2
        
        # Find LT1 index (crossover point where anaerobic begins to contribute)
        lt1_indices = np.where(np.diff(np.sign(vlass - available_pyruvate)) > 0)[0]
        lt1_index = lt1_indices[0] if len(lt1_indices) > 0 else at_index // 2
        
        # Find FatMax index (point of maximum fat utilization)
        fatmax_index = np.argmax(available_pyruvate)
        
        # Calculate percentages of VO2max
        at_percent = (vo2ss[at_index] / self.vo2max) * 100
        lt1_percent = (vo2ss[lt1_index] / self.vo2max) * 100
        fatmax_percent = (vo2ss[fatmax_index] / self.vo2max) * 100
        
        # Calculate power values at thresholds
        at_power = power[at_index]
        lt1_power = power[lt1_index]
        fatmax_power = power[fatmax_index]
        
        return {
            'at_index': at_index,
            'lt1_index': lt1_index,
            'fatmax_index': fatmax_index,
            'at_percent': at_percent,
            'lt1_percent': lt1_percent,
            'fatmax_percent': fatmax_percent,
            'at_power': at_power,
            'lt1_power': lt1_power,
            'fatmax_power': fatmax_power
        }
    
    def calculate_steady_state_lactate(self, kel=4):
        vo2ss, vlass, vlanet, available_pyruvate, la_comb = self.calculate_lactate_dynamics()
        thresholds = self.find_thresholds()
        at_index = thresholds['at_index']
        
        # Only calculate below the threshold
        vo2ss_below = vo2ss[:at_index+1]
        
        # Calculate steady state lactate concentration
        denominator = ((self.lactate_combustion_factor / self.VolRel) * vo2ss_below) * (1 + (self.Ks2 / ((self.Ks1 * vo2ss_below) / (self.vo2max - vo2ss_below)) ** (3 / 2))) - (self.vlamax * 60)
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-9)
        
        class_below = np.sqrt((self.vlamax * kel * 60) / denominator)
        
        # For values above threshold, create simple linear increase
        class_above = np.linspace(class_below[-1], class_below[-1] * 3, len(vo2ss) - len(class_below))
        
        # Combine arrays
        class_total = np.concatenate([class_below, class_above])
        
        return class_total
    
    def calculate_macronutrient_utilization(self):
        vo2ss, vlass, vlanet, available_pyruvate, la_comb = self.calculate_lactate_dynamics()
        thresholds = self.find_thresholds()
        at_index = thresholds['at_index']
        
        # Calculate carbohydrate utilization (g/h)
        cho_util = vlass * (self.body_mass * self.VolRel) * 60 / 1000 / 2 * 162.14
        
        # Calculate fat utilization (g/h)
        # Only valid below AT
        fat_util_below = vlanet[:at_index+1] * self.VolRel / self.lactate_combustion_factor * self.body_mass * 60 * 4.65 / 9.5 / 1000
        
        # For points above AT, decrease fat utilization
        if at_index < len(vo2ss) - 1:
            fat_util_above = np.linspace(fat_util_below[-1], fat_util_below[-1] * 0.5, len(vo2ss) - (at_index + 1))
            fat_util = np.concatenate([fat_util_below, fat_util_above])
        else:
            fat_util = fat_util_below
        
        # Calculate total energy utilization (kcal/h)
        energy_cho = cho_util * 4  # 4 kcal per gram of carbs
        energy_fat = fat_util * 9  # 9 kcal per gram of fat
        total_energy = energy_cho + energy_fat
        
        return cho_util, fat_util, total_energy
    
    def calculate_glycogen_storage(self, body_fat_percentage=14):
        # Estimate fitness level based on VO2max and VLaMax
        fitness_level = estimate_fitness_level(self.vo2max, self.vlamax)
        
        # Calculate glycogen storage
        glycogen_storage = estimate_glycogen_storage(self.body_mass, body_fat_percentage, fitness_level)
        
        return glycogen_storage, fitness_level
    
    def classify_performance(self):
        thresholds = self.find_thresholds()
        
        # Classify based on thresholds and VLaMax
        category = classify_efficiency(
            self.vlamax, 
            thresholds['at_percent'], 
            thresholds['lt1_percent'], 
            thresholds['fatmax_percent']
        )
        
        return category
    
    def plot_lactate_dynamics(self, fig_size=(12, 8)):
        vo2ss, vlass, vlanet, available_pyruvate, la_comb = self.calculate_lactate_dynamics()
        thresholds = self.find_thresholds()
        intensity, power, overall_demand = self.calculate_intensity()
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Plot lactate formation and combustion
        ax1.plot(intensity, vlass, label='Gross Lactate Production', color=COLORS['gross_lactate'], linewidth=2)
        ax1.plot(intensity, la_comb, label='Lactate Combustion', color=COLORS['lactate_combustion'], linewidth=2)
        ax1.plot(intensity, vlanet, label='Net Lactate', color=COLORS['net_lactate'], linewidth=2)
        ax1.plot(intensity, available_pyruvate, label='Available Pyruvate', color=COLORS['available_pyruvate'], linewidth=2)
        
        # Mark thresholds
        ax1.axvline(x=intensity[thresholds['at_index']], color=COLORS['at'], linestyle='--', linewidth=1.5, label=f'AT ({thresholds["at_percent"]:.1f}% VO2max)')
        ax1.axvline(x=intensity[thresholds['lt1_index']], color=COLORS['lt1'], linestyle='--', linewidth=1.5, label=f'LT1 ({thresholds["lt1_percent"]:.1f}% VO2max)')
        ax1.axvline(x=intensity[thresholds['fatmax_index']], color=COLORS['fatmax'], linestyle='--', linewidth=1.5, label=f'FatMax ({thresholds["fatmax_percent"]:.1f}% VO2max)')
        
        # Set axes labels and title
        ax1.set_xlabel('% of VO2max', fontweight='bold')
        ax1.set_ylabel('mmol/L/min', fontweight='bold')
        ax1.set_title('Lactate Dynamics Model', fontweight='bold', fontsize=14)
        
        # Add second x-axis for power
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(intensity[::10])
        ax2.set_xticklabels([f"{p:.0f}" for p in power[::10]])
        ax2.set_xlabel('Power (watts)', fontweight='bold')
        
        # Add grid and legend
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', frameon=True)
        
        plt.tight_layout()
        return fig
    
    def plot_fuel_utilization(self, fig_size=(12, 8)):
        cho_util, fat_util, total_energy = self.calculate_macronutrient_utilization()
        intensity, power, overall_demand = self.calculate_intensity()
        thresholds = self.find_thresholds()
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Plot carbohydrate and fat utilization
        ax1.plot(intensity, cho_util, label='Carbohydrate (g/h)', color='#FF6B6B', linewidth=2)
        ax1.plot(intensity, fat_util, label='Fat (g/h)', color='#4ECDC4', linewidth=2)
        
        # Mark thresholds
        ax1.axvline(x=intensity[thresholds['at_index']], color=COLORS['at'], linestyle='--', linewidth=1.5, label=f'AT ({thresholds["at_percent"]:.1f}% VO2max)')
        ax1.axvline(x=intensity[thresholds['fatmax_index']], color=COLORS['fatmax'], linestyle='--', linewidth=1.5, label=f'FatMax ({thresholds["fatmax_percent"]:.1f}% VO2max)')
        
        # Set axes labels and title
        ax1.set_xlabel('% of VO2max', fontweight='bold')
        ax1.set_ylabel('Fuel Utilization (g/h)', fontweight='bold')
        ax1.set_title('Substrate Utilization', fontweight='bold', fontsize=14)
        
        # Add second y-axis for total energy
        ax3 = ax1.twinx()
        ax3.plot(intensity, total_energy, label='Total Energy (kcal/h)', color='#556270', linewidth=2, linestyle=':')
        ax3.set_ylabel('Energy Expenditure (kcal/h)', fontweight='bold')
        
        # Add second x-axis for power
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(intensity[::10])
        ax2.set_xticklabels([f"{p:.0f}" for p in power[::10]])
        ax2.set_xlabel('Power (watts)', fontweight='bold')
        
        # Add grid and legend
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper left', frameon=True)
        
        plt.tight_layout()
        return fig
    
    def plot_lactate_concentration(self, fig_size=(12, 8)):
        intensity, power, overall_demand = self.calculate_intensity()
        lactate_conc = self.calculate_steady_state_lactate()
        thresholds = self.find_thresholds()
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=fig_size)
        
        # Plot lactate concentration
        ax1.plot(intensity, lactate_conc, label='Lactate Concentration', color='#FF6B6B', linewidth=2)
        
        # Mark thresholds
        ax1.axvline(x=intensity[thresholds['at_index']], color=COLORS['at'], linestyle='--', linewidth=1.5, label=f'AT ({thresholds["at_percent"]:.1f}% VO2max)')
        ax1.axvline(x=intensity[thresholds['lt1_index']], color=COLORS['lt1'], linestyle='--', linewidth=1.5, label=f'LT1 ({thresholds["lt1_percent"]:.1f}% VO2max)')
        
        # Set axes labels and title
        ax1.set_xlabel('% of VO2max', fontweight='bold')
        ax1.set_ylabel('Blood Lactate (mmol/L)', fontweight='bold')
        ax1.set_title('Steady State Lactate Concentration', fontweight='bold', fontsize=14)
        
        # Add second x-axis for power
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(intensity[::10])
        ax2.set_xticklabels([f"{p:.0f}" for p in power[::10]])
        ax2.set_xlabel('Power (watts)', fontweight='bold')
        
        # Add grid and legend
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', frameon=True)
        
        plt.tight_layout()
        return fig

# Streamlit Application
def streamlit_app():
    st.set_page_config(page_title="Metabolic Analyzer", layout="wide")
    
    st.title("Metabolic Analyzer")
    st.markdown("Based on Mader's Metabolic Model")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Athlete Data")
        
        # Basic metrics
        vo2max = st.number_input("VO2max (ml/kg/min)", min_value=30.0, max_value=90.0, value=60.0, step=0.1)
        body_mass = st.number_input("Body Mass (kg)", min_value=40.0, max_value=150.0, value=78.0, step=0.1)
        bmi = st.number_input("BMI", min_value=16.0, max_value=35.0, value=22.0, step=0.1)
        age = st.number_input("Age", min_value=18, max_value=80, value=38, step=1)
        
        st.header("Power Data")
        # Power metrics
        vo2max_power = st.number_input("VO2max Power (watts)", min_value=100, max_value=600, value=370, step=5)
        sprint_power = st.number_input("20s Sprint Power (watts)", min_value=300, max_value=2000, value=776, step=10)
        workload_3min = st.number_input("3-min Test Power (watts)", min_value=0, max_value=500, value=0, step=5)
        workload_6min = st.number_input("6-min Test Power (watts)", min_value=0, max_value=500, value=0, step=5)
        
        st.header("Advanced Parameters")
        # Advanced metrics (with explanations)
        vlamax = st.number_input("VLaMax (calculate automatically if 0)", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
        st.caption("Maximum rate of glycolysis (mmol/L/s)")
        
        vol_rel = st.number_input("VolRel (calculate automatically if 0)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        st.caption("Relative distribution volume of lactate")
        
        lactate_combustion_factor = st.number_input("Lactate Combustion Factor", min_value=0.01, max_value=0.02, value=0.01576, step=0.0001, format="%.5f")
        st.caption("Rate of lactate oxidation per ml of O2")
        
        calculate_button = st.button("Calculate Metrics")
    
    # Main content area
    if calculate_button:
        # Create analyzer object
        analyzer = MetabolicAnalyzer(
            vo2max=vo2max,
            body_mass=body_mass,
            workload_3min=workload_3min,
            workload_6min=workload_6min,
            vlamax=vlamax,
            vol_rel=vol_rel,
            sprint_power=sprint_power,
            vo2max_power=vo2max_power,
            bmi=bmi,
            age=age,
            lactate_combustion_factor=lactate_combustion_factor
        )
        
        # Calculate thresholds
        thresholds = analyzer.find_thresholds()
        category = analyzer.classify_performance()
        glycogen_storage, fitness_level = analyzer.calculate_glycogen_storage()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Metabolic Parameters")
            st.write(f"VLaMax: {analyzer.vlamax:.4f} mmol/L/s")
            st.write(f"VolRel: {analyzer.VolRel:.4f}")
            st.write(f"Fitness Level: {fitness_level.title()}")
            
        with col2:
            st.subheader("Threshold Powers")
            st.write(f"FatMax: {thresholds['fatmax_power']:.1f} watts ({thresholds['fatmax_percent']:.1f}% VO2max)")
            st.write(f"LT1: {thresholds['lt1_power']:.1f} watts ({thresholds['lt1_percent']:.1f}% VO2max)")
            st.write(f"AT: {thresholds['at_power']:.1f} watts ({thresholds['at_percent']:.1f}% VO2max)")
            
        with col3:
            st.subheader("Performance Analysis")
            st.write(f"Metabolic Profile: {category}")
            st.write(f"Glycogen Storage: {glycogen_storage:.1f} g")
        
        # Display charts
        st.header("Metabolic Charts")
        
        tab1, tab2, tab3 = st.tabs(["Lactate Dynamics", "Fuel Utilization", "Lactate Concentration"])
        
        with tab1:
            lactate_fig = analyzer.plot_lactate_dynamics()
            st.pyplot(lactate_fig)
            
        with tab2:
            fuel_fig = analyzer.plot_fuel_utilization()
            st.pyplot(fuel_fig)
            
        with tab3:
            lactate_conc_fig = analyzer.plot_lactate_concentration()
            st.pyplot(lactate_conc_fig)
        
        # Training recommendations based on metabolic profile
        st.header("Training Recommendations")
        
        # Create recommendation zones
        zones_df = pd.DataFrame({
            "Zone": ["Recovery", "Endurance/FatMax", "Tempo/LT1", "Threshold/AT", "VO2max", "Anaerobic"],
            "Lower Power": [0, thresholds['fatmax_power']*0.8, thresholds['lt1_power']*0.9, thresholds['at_power']*0.9, thresholds['at_power']*1.05, thresholds['at_power']*1.2],
            "Upper Power": [thresholds['fatmax_power']*0.8, thresholds['lt1_power']*0.9, thresholds['at_power']*0.9, thresholds['at_power']*1.05, thresholds['at_power']*1.2, vo2max_power],
            "Description": [
                "Active recovery, very low intensity",
                "Fat oxidation maximized, long endurance training",
                "Moderate intensity, steady efforts",
                "Lactate threshold training, medium intervals",
                "High intensity interval training",
                "Anaerobic power development, short intervals"
            ]
        })
        
        st.table(zones_df)
        
        # Custom recommendations based on profile
        st.subheader("Based on Your Metabolic Profile")
        
        if "Endurance" in category:
            st.write("Your profile shows good endurance characteristics. Focus on maintaining your aerobic base while incorporating some high-intensity intervals to improve VLaMax if needed.")
        elif "Power" in category:
            st.write("Your profile shows stronger anaerobic characteristics. Focus on building your aerobic base with longer rides in the FatMax zone to improve fat oxidation and endurance.")
        elif "Mixed" in category:
            st.write("Your profile shows a balanced mix of aerobic and anaerobic capabilities. You can work on both systems effectively.")
        
        # Training focus recommendations based on thresholds
        if thresholds['at_percent'] < 70:
            st.write("Priority: Increase your anaerobic threshold by doing sweet spot (88-95% of threshold) and threshold intervals.")
        if thresholds['fatmax_percent'] < 55:
            st.write("Priority: Improve fat metabolism by doing more long, steady rides in the endurance zone.")
        
        # Download data option
        csv_data = pd.DataFrame({
            "Metric": ["VLaMax", "VolRel", "FatMax Power", "FatMax %VO2max", "LT1 Power", "LT1 %VO2max", "AT Power", "AT %VO2max", "Fitness Level", "Metabolic Profile", "Glycogen Storage"],
            "Value": [analyzer.vlamax, analyzer.VolRel, thresholds['fatmax_power'], thresholds['fatmax_percent'], thresholds['lt1_power'], thresholds['lt1_percent'], thresholds['at_power'], thresholds['at_percent'], fitness_level, category, glycogen_storage]
        })
        
        st.download_button(
            label="Download Results as CSV",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name='metabolic_analysis.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    streamlit_app()
