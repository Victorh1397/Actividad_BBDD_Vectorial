"""Query workloads, graded judgments, listings and events (RF-16, RF-17, RF-19)."""

from __future__ import annotations

import pytest

from aurum_market.config import RELEVANCE_MAPPING, RELEVANCE_THRESHOLD
from aurum_market.data import (
    DataIntegrityError,
    load_catalog_events,
    load_development_queries,
    load_evaluation_queries,
    load_filtered_queries,
    load_incoming_listings,
    load_relevance_judgments,
)


class TestDevelopmentQueries:
    def test_loads_the_eight_declared_queries(self) -> None:
        queries = load_development_queries()
        assert len(queries) == 8

    def test_keeps_both_identifiers(self) -> None:
        """El workload_id es la clave; el numérico permite unir con los qrels."""
        query = load_development_queries()[0]
        assert query.query_id == "DEV-13357"
        assert query.numeric_id == 13357
        assert query.text == "base tapizada 160x200 sin patas"

    def test_query_ids_are_unique(self) -> None:
        queries = load_development_queries()
        assert len({query.query_id for query in queries}) == len(queries)


class TestEvaluationQueries:
    def test_loads_the_twelve_blind_queries(self) -> None:
        assert len(load_evaluation_queries()) == 12

    def test_covers_four_intents_in_three_formulations(self) -> None:
        """Las tres formulaciones comprueban estabilidad ante el léxico."""
        queries = load_evaluation_queries()
        types = {query.query_type for query in queries}
        assert types == {"context", "direct", "semantic"}
        intents = {query.query_id.rsplit("-", 1)[0] for query in queries}
        assert len(intents) == 4

    def test_ids_match_the_artifact_contract(self) -> None:
        """resultados_busqueda.csv exige este patrón de evaluation_id."""
        for query in load_evaluation_queries():
            assert query.query_id.startswith("EVAL-")
            assert query.query_id.rsplit("-", 1)[1] in {"context", "direct", "semantic"}


class TestFilteredQueries:
    def test_loads_the_four_brand_constrained_queries(self) -> None:
        queries = load_filtered_queries()
        assert len(queries) == 4
        assert all(query.brand for query in queries)

    def test_brands_are_the_declared_ones(self) -> None:
        brands = {query.brand for query in load_filtered_queries()}
        assert brands == {"Einhell", "Apple", "NIKE", "SAMSUNG"}

    def test_an_unsupported_constraint_fails_loudly(self, tmp_path) -> None:
        """Ignorar un filtro desconocido convertiría la búsqueda en global."""
        csv = tmp_path / "consultas_filtradas.csv"
        csv.write_text(
            "workload_id,query_text,filter_field,filter_operator,filter_value,"
            "expected_property\n"
            "FILTER-001,taladro,color,contains,Rojo,x\n",
            encoding="utf-8",
        )
        with pytest.raises(DataIntegrityError, match="no soportada"):
            load_filtered_queries(data_directory=tmp_path)


class TestRelevanceJudgments:
    def test_covers_every_development_query(self) -> None:
        judgments = load_relevance_judgments()
        expected = {query.query_id for query in load_development_queries()}
        assert set(judgments) == expected

    def test_qrels_join_has_no_orphans(self) -> None:
        """248 juicios repartidos entre las 8 consultas, sin quedarse ninguno."""
        judgments = load_relevance_judgments()
        assert sum(len(items) for items in judgments.values()) == 248

    def test_grades_follow_the_statement_mapping(self) -> None:
        judgments = load_relevance_judgments()
        grades = {value for items in judgments.values() for value in items.values()}
        assert grades <= set(RELEVANCE_MAPPING.values())

    def test_a_regraded_label_is_rejected(self, tmp_path) -> None:
        """P-05: la escala no puede cambiar en silencio entre experimentos."""
        (tmp_path / "consultas_desarrollo.csv").write_text(
            "workload_id,query_id,query_text,query_type\n"
            "DEV-13357,13357,base tapizada,customer_query\n",
            encoding="utf-8",
        )
        (tmp_path / "relevancias_desarrollo.csv").write_text(
            "query_id,product_id,esci_label,relevance\n13357,B001,E,2\n",
            encoding="utf-8",
        )
        with pytest.raises(DataIntegrityError, match="enunciado fija"):
            load_relevance_judgments(data_directory=tmp_path)

    def test_recall_at_10_has_a_structural_ceiling(self) -> None:
        """198 relevantes en 8 consultas y solo 10 posiciones: el techo es 0,519.

        No es un defecto del sistema, es aritmética. Fijar el valor medido
        impide que el informe cite una estimación en vez del dato.
        """
        from statistics import fmean

        from aurum_market.evaluation.metrics import recall_ceiling_at_k

        judgments = load_relevance_judgments()
        relevant = sum(
            1
            for items in judgments.values()
            for value in items.values()
            if value >= RELEVANCE_THRESHOLD
        )
        assert relevant == 198

        ceilings = {
            query_id: recall_ceiling_at_k(items, k=10)
            for query_id, items in judgments.items()
        }
        assert fmean(ceilings.values()) == pytest.approx(0.519, abs=1e-3)
        # La dispersión importa tanto como la media: el mismo Recall@10 significa
        # cosas opuestas en la consulta más fácil y en la más difícil.
        assert min(ceilings.values()) == pytest.approx(10 / 39, abs=1e-3)
        assert max(ceilings.values()) == pytest.approx(1.0)


class TestIncomingListings:
    def test_development_split_is_labelled_and_balanced(self) -> None:
        listings = load_incoming_listings("desarrollo")
        assert len(listings) == 14
        duplicates = [item for item in listings if item.is_duplicate]
        assert len(duplicates) == 7
        assert all(item.is_labelled for item in listings)

    def test_every_labelled_duplicate_names_its_reference(self) -> None:
        for listing in load_incoming_listings("desarrollo"):
            if listing.is_duplicate:
                assert listing.reference_product_id
            else:
                assert listing.reference_product_id is None

    def test_evaluation_split_carries_no_labels(self) -> None:
        """P-04: el conjunto ciego no puede filtrar información a la calibración."""
        listings = load_incoming_listings("evaluacion")
        assert len(listings) == 14
        assert all(item.is_duplicate is None for item in listings)
        assert all(item.reference_product_id is None for item in listings)

    def test_an_unknown_split_is_rejected(self) -> None:
        with pytest.raises(DataIntegrityError, match="Split desconocido"):
            load_incoming_listings("profesorado")  # type: ignore[arg-type]


class TestCatalogEvents:
    def test_loads_the_twenty_four_declared_events(self) -> None:
        assert len(load_catalog_events()) == 24

    def test_events_arrive_ordered_by_sequence(self) -> None:
        events = load_catalog_events()
        assert [event.sequence for event in events] == list(range(1, 25))

    def test_operations_split_into_upserts_and_deletes(self) -> None:
        events = load_catalog_events()
        upserts = [e for e in events if e.operation == "UPSERT"]
        deletes = [e for e in events if e.operation == "DELETE"]
        assert len(upserts) == 16
        assert len(deletes) == 8

    def test_upserts_carry_a_sheet_and_deletions_do_not(self) -> None:
        for event in load_catalog_events():
            assert event.record_id
            assert event.product_id
            if event.is_deletion:
                assert event.record is None
            else:
                assert event.require_record().title

    def test_the_eight_deletions_only_identify_their_target(self) -> None:
        """Las bajas llegan sin título ni texto: no hay ficha que describir."""
        deletions = [event for event in load_catalog_events() if event.is_deletion]
        assert len(deletions) == 8
        assert all(event.record is None for event in deletions)

    def test_a_gap_in_the_sequence_is_rejected(self, tmp_path) -> None:
        """Aplicar eventos con huecos dejaría un estado final impredecible."""
        header = (
            "sequence,event_id,operation,record_id,product_id,title,brand,color,"
            "locale,text,catalog_version,active\n"
        )
        rows = (
            "1,EVT-001,UPSERT,000bd6e8-a995-56d0-ba03-559885ccef39,B0818K237B,"
            "Vestido,Marca,Negro,es,Vestido,2,True\n"
            "3,EVT-003,DELETE,0037a9df-8492-508f-8167-c09624801216,B086YX9RK5,"
            "IQOS,IQOS,,es,IQOS,1,False\n"
        )
        (tmp_path / "eventos_catalogo.csv").write_text(header + rows, encoding="utf-8")
        with pytest.raises(DataIntegrityError, match="sin huecos"):
            load_catalog_events(data_directory=tmp_path)


class TestAssumptionsBehindTheExecutionOrder:
    """Los supuestos de [ADR-001], blindados con tests.

    Si alguno de estos dejara de pasar, el orden canónico de ejecución habría
    dejado de estar justificado y el ADR tendría que revisarse otra vez.
    """

    def test_events_target_the_duplicate_reference_products(self) -> None:
        """Las 7 referencias de desarrollo están todas tocadas por eventos."""
        mutated = {event.product_id for event in load_catalog_events()}
        references = {
            listing.reference_product_id
            for listing in load_incoming_listings("desarrollo")
            if listing.reference_product_id
        }
        assert references <= mutated, (
            "ADR-001 se apoya en que los eventos tocan TODAS las referencias"
        )

    def test_development_references_are_updated_never_deleted(self) -> None:
        """La asimetría clave: en desarrollo se actualizan, no se borran.

        Si se borraran, no habría forma de calibrar el umbral con casos
        positivos: el candidato no existiría.
        """
        deleted = {
            event.product_id for event in load_catalog_events() if event.is_deletion
        }
        references = {
            listing.reference_product_id
            for listing in load_incoming_listings("desarrollo")
            if listing.reference_product_id
        }
        assert not (references & deleted)

    def test_judgments_ignore_the_products_that_events_add(self) -> None:
        """El argumento central de ADR-001.

        Los AURUM-NEW-* no figuran en ningún juicio de relevancia, así que las
        métricas de ranking se calculan sobre el catálogo base y los eventos no
        pertenecen al recorrido de medición.
        """
        judged = {
            product_id
            for items in load_relevance_judgments().values()
            for product_id in items
        }
        added = {
            event.product_id
            for event in load_catalog_events()
            if not event.is_deletion and event.product_id.startswith("AURUM-NEW")
        }
        assert len(added) == 8, "se esperaban 8 altas sintéticas en los eventos"
        assert not (judged & added)

    @pytest.mark.slow
    def test_every_judged_product_exists_in_the_base_catalog(self) -> None:
        """248 juicios, cero huérfanos: la verdad de referencia es el catálogo base."""
        from aurum_market.data import load_catalog

        catalog = load_catalog("full")
        judged = {
            product_id
            for items in load_relevance_judgments().values()
            for product_id in items
        }
        assert judged <= set(catalog.by_product_id)

    @pytest.mark.slow
    def test_deletions_remove_products_that_really_exist(self) -> None:
        """Una baja sobre un producto inexistente no probaría nada."""
        from aurum_market.data import load_catalog

        catalog = load_catalog("full")
        deleted = {
            event.product_id for event in load_catalog_events() if event.is_deletion
        }
        assert deleted <= set(catalog.by_product_id)
