import concurrent
import datetime as dt
import itertools
import sys
import traceback
import zoneinfo
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cantera as ct
from tqdm import tqdm

from funcs.dir import MECH_DIR
from funcs.simulation.cell_size import CvConfig, calculate_westbrook_only
from funcs.simulation.thermo import diluted_species_dict, match_adiabatic_temp
from sdtoolbox.output import clear_simulation_database
from sdtoolbox.postshock import CJspeed

if TYPE_CHECKING:
    from concurrent.futures import Future

# noinspection PyUnresolvedReferences
ct.add_directory(MECH_DIR)

PHI = 1
FUEL = "CH4"
OXIDIZER = "N2O"
P_0 = 101325
T_0 = 300

CONN_INFO = "postgresql://postgres@localhost:5432/perturbation_study"


@dataclass(frozen=True)
class Perturbation:
    rxn_no: int
    fraction: float


@dataclass(frozen=True)
class SimulationInputs:
    diluent: str | None
    dil_mf: float
    dil_condition: str
    mech: str
    perturbation: Perturbation | None
    species: set[str]

    def error_header(self) -> str:
        # noinspection PyTypeChecker
        self_dict = asdict(self)
        max_key_len = 0
        max_val_len = 0
        for k, v in self_dict.items():
            max_key_len = max(max_key_len, len(k))
            max_val_len = max(max_val_len, len(str(v)))

        l_r_border_len = 4  # 2 * (space + equals)
        space_len = 1
        colon_len = 1
        border = "=" * (max_key_len + max_val_len + l_r_border_len + space_len + colon_len)
        lines = [
            border,
            *(
                f"= {f"{k}:".ljust(max_key_len + colon_len)} {str(v).ljust(max_val_len)} ="
                for k, v in self_dict.items()
            ),
            border,
        ]
        return "\n".join(lines)


def simulate_single_condition(inputs: SimulationInputs) -> None:
    gas = ct.Solution(inputs.mech)
    gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)
    if inputs.diluent is not None:
        q = diluted_species_dict(gas.mole_fraction_dict(), inputs.diluent, inputs.dil_mf)
    else:
        q = gas.mole_fraction_dict()
    cj_speed = CJspeed(P_0, T_0, q, inputs.mech)
    if inputs.perturbation is None:
        perturbed_reaction = None
        perturbation_fraction = 0
    else:
        perturbed_reaction = inputs.perturbation.rxn_no
        perturbation_fraction = inputs.perturbation.fraction
    calculate_westbrook_only(
        mechanism=inputs.mech,
        initial_temp=T_0,
        initial_press=P_0,
        fuel=FUEL,
        oxidizer=OXIDIZER,
        equivalence=PHI,
        phi_nom=PHI,
        cv_config=CvConfig(max_tries=3, max_step=1e-6, end_time=12e-6, solver_method="Radau"),
        diluent=inputs.diluent,
        match="tad",
        diluent_mol_frac=inputs.dil_mf,
        dil_condition=inputs.dil_condition,
        cj_speed=cj_speed,
        perturbed_reaction=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
        spec_indices=[i for i, spec in enumerate(gas.species()) if spec.name in inputs.species],
        rxn_indices=list(range(len(gas.reactions()))) if perturbed_reaction is None else [perturbed_reaction],
        conninfo=CONN_INFO,
    )


def write_error_log(error_log: Path, inputs: SimulationInputs, err: BaseException) -> None:
    with error_log.open("a") as f:
        f.write(f"{inputs.error_header()}\n{''.join(traceback.format_exception(err))}\n\n")


def handle_result(result: "Future", inputs: SimulationInputs, error_log: Path, counter: tqdm, n_errors: int) -> int:
    if (err := result.exception()) is not None:
        n_errors += 1
        write_error_log(error_log, inputs, err)
        counter.colour = "red"
        counter.set_postfix_str(f"{n_errors} ERRORS")
    counter.update()
    return n_errors


def run_simulations(output_base: Path, simulation_inputs: list[SimulationInputs]) -> None:
    error_log = output_base.with_stem(f"{output_base.stem}_errors").with_suffix("")
    if error_log.exists():
        error_log.unlink()
    futures = {}
    n_errors = 0
    with (
        ProcessPoolExecutor() as executor,
        tqdm(total=len(simulation_inputs), unit="calc", file=sys.stdout, colour="green", desc="Running") as counter,
    ):
        for inputs in simulation_inputs:
            f = executor.submit(simulate_single_condition, inputs)
            futures[f] = inputs

        for result in concurrent.futures.as_completed(futures):
            n_errors = handle_result(result, futures[result], error_log, counter, n_errors)

        counter.set_description_str("Done")


def main() -> None:
    clear_simulation_database(CONN_INFO)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    local_tz = zoneinfo.ZoneInfo("US/Pacific")
    today = dt.datetime.now(local_tz).date().isoformat()

    mech = "gri30_highT_inerts.yaml"
    output_base = output_dir / f"perturbation_study_{today}_{mech}"

    gas = ct.Solution(mech)

    rxn_pert_frac = {
        7: -0.025,
        77: -0.025,
        86: -0.025,
        180: -0.025,
        281: -0.025,
        249: -0.025,
        237: -0.025,
        298: -0.025,
    }

    equations_to_track: list[int | None] = [None, *range(len(gas.reactions()))]
    equations_in_top_5: set[int] = {
        # # N2 RCC high dilution
        # *(108, 88, 184, 41, 242),
        # # N2 RCC low dilution
        # *(289, 257, 184, 88, 242),
        # # CO2 RCC high dilution
        # *(154, 88, 257, 184, 242),
        # # CO2 RCC low dilution
        # *(154, 289, 88, 257, 184),
        # N2 mul high dilution
        *(157, 52, 158, 182, 184),
        # N2 mul low dilution
        *(157, 52, 158, 182, 184),
        # CO2 mul high dilution
        *(157, 52, 158, 182, 184),
        # CO2 mul low dilution
        *(157, 52, 158, 182, 184),
    }
    species_in_top_5: set[str] = set(
        itertools.chain.from_iterable(
            {*gas.reaction(idx).reactants.keys(), *gas.reaction(idx).products.keys()}
            for idx in equations_in_top_5
        )
    )
    simulation_inputs = []
    co2_low = 0.1
    co2_high = 0.2
    n2_low = match_adiabatic_temp(
        mech=mech,
        fuel=FUEL,
        oxidizer=OXIDIZER,
        phi=PHI,
        dil_original="CO2",
        dil_original_mol_frac=co2_low,
        dil_new="N2",
        init_temp=T_0,
        init_press=P_0,
    )
    n2_high = match_adiabatic_temp(
        mech=mech,
        fuel=FUEL,
        oxidizer=OXIDIZER,
        phi=PHI,
        dil_original="CO2",
        dil_original_mol_frac=co2_high,
        dil_new="N2",
        init_temp=T_0,
        init_press=P_0,
    )
    for diluent, dil_mf, dil_condition in (
        (None, 0, "undiluted"),
        ("CO2", co2_low, "low"),
        ("CO2", co2_high, "high"),
        ("CO2i", co2_low, "low"),
        ("CO2i", co2_high, "high"),
        ("N2", n2_low, "low"),
        ("N2", n2_high, "high"),
        ("N2i", n2_low, "low"),
        ("N2i", n2_high, "high"),
    ):
        for rxn_no in equations_to_track:
            # these are finicky
            perturbation_fraction = rxn_pert_frac.get(rxn_no, -0.05)
            simulation_inputs.append(
                SimulationInputs(
                    diluent=diluent,
                    dil_mf=dil_mf,
                    dil_condition=dil_condition,
                    mech=mech,
                    perturbation=(
                        None if rxn_no is None else Perturbation(rxn_no=rxn_no, fraction=perturbation_fraction)
                    ),
                    species={"H", "OH", "NO", "CO2", "N2"}.union(species_in_top_5)
                )
            )

    run_simulations(output_base, simulation_inputs)


if __name__ == "__main__":
    main()
