"""Testes de reassemble.py."""

import unittest


class ComputeSplitPointsTests(unittest.TestCase):
    def _balloon(self, y_top, y_bottom):
        from strip.types import Balloon, BBox
        return Balloon(strip_bbox=BBox(0, y_top, 100, y_bottom), confidence=0.9)

    def test_split_points_for_empty_strip(self):
        from strip.reassemble import _compute_split_points
        points = _compute_split_points(strip_height=0, balloons=[], target_count=60)
        self.assertEqual(points, [0, 0])

    def test_split_points_with_no_balloons_distributes_evenly(self):
        from strip.reassemble import _compute_split_points
        points = _compute_split_points(strip_height=6000, balloons=[], target_count=60)
        # Sem balões, distribui em 60 fatias de ~100 px
        self.assertEqual(len(points), 61)
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], 6000)
        # Step médio próximo de 100
        steps = [points[i + 1] - points[i] for i in range(len(points) - 1)]
        self.assertAlmostEqual(sum(steps) / len(steps), 100, delta=2)

    def test_split_points_snap_above_balloon(self):
        from strip.reassemble import _compute_split_points
        # Strip de 1000 px, target=2 -> ideal split em y=500
        # Mas tem balão em y=480..520 -> deve subir o split pra y=478 (480-2)
        points = _compute_split_points(
            strip_height=1000,
            balloons=[self._balloon(480, 520)],
            target_count=2,
        )
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], 1000)
        # split intermediário não pode estar dentro do balão
        for split_y in points[1:-1]:
            self.assertFalse(480 <= split_y <= 520, f"split em {split_y} cortou balão 480-520")

    def test_split_points_handle_balloon_at_split_boundary_walks_up_until_safe(self):
        from strip.reassemble import _compute_split_points
        # 3 balões empilhados que cobrem y=400..900. Apenas um split intermediário possível: y < 400.
        balloons = [
            self._balloon(400, 500),
            self._balloon(510, 700),
            self._balloon(710, 900),
        ]
        points = _compute_split_points(strip_height=1000, balloons=balloons, target_count=2)
        # Split deve ficar antes de y=400
        self.assertLess(points[1], 400)


class SplitPointMarginTests(unittest.TestCase):
    """Testa que os split points ficam pelo menos 12px acima do top do balao."""

    def _balloon(self, y_top, y_bottom):
        from strip.types import Balloon, BBox
        return Balloon(strip_bbox=BBox(0, y_top, 100, y_bottom), confidence=0.9)

    def test_split_points_leave_at_least_12px_margin_above_balloon(self):
        """Split deve ficar >= 12 px acima do topo do balao (margem de seguranca)."""
        from strip.reassemble import _compute_split_points
        balloons = [self._balloon(480, 520)]
        points = _compute_split_points(strip_height=1000, balloons=balloons, target_count=2)
        for p in points[1:-1]:
            if abs(p - 480) < 100:
                # Se o split tentou cair perto deste balao, deve estar >= 12px antes do top
                self.assertLessEqual(p, 480 - 12,
                    f"Split em {p} esta a apenas {480 - p}px do topo do balao (minimo 12px)")

    def test_split_points_no_tiny_pages(self):
        """Nenhuma pagina deve ter menos de 50px apos o split."""
        from strip.reassemble import _compute_split_points
        # Baloes proximos do topo forcam splits pra cima
        balloons = [self._balloon(i * 40, i * 40 + 30) for i in range(1, 5)]
        points = _compute_split_points(strip_height=2000, balloons=balloons, target_count=10)
        sizes = [points[i + 1] - points[i] for i in range(len(points) - 1)]
        self.assertTrue(all(s >= 50 for s in sizes),
            f"Pagina muito pequena encontrada: {min(sizes)}px em {sizes}")

    def test_visual_padding_prevents_split_inside_balloon_spikes(self):
        """padding=20 faz o algoritmo tratar o balao como 20px maior para evitar spikes."""
        from strip.reassemble import _compute_split_points
        # Balao em 480..520 com padding=20 -> espaco proibido = 460..540
        # Split em 500 deve subir para <= 460
        balloons = [self._balloon(480, 520)]
        points = _compute_split_points(
            strip_height=1000, balloons=balloons, target_count=2,
            balloon_visual_padding=20,
        )
        for p in points[1:-1]:
            if 450 < p < 560:
                self.assertLessEqual(p, 460,
                    f"Com padding=20, split em {p} esta dentro da zona de exclusao [460..540]")


class AssembleOutputPagesTests(unittest.TestCase):
    def test_assemble_concatenated_outputs_equal_strip_image(self):
        from strip.reassemble import assemble_output_pages
        from strip.types import VerticalStrip
        import numpy as np

        rng = np.random.default_rng(0)
        strip_img = rng.integers(0, 256, (1000, 300, 3), dtype=np.uint8)
        strip = VerticalStrip(image=strip_img, width=300, height=1000, source_page_breaks=[0, 1000])

        output_pages = assemble_output_pages(strip, balloons=[], target_count=10)
        self.assertEqual(len(output_pages), 10)

        # Concatenar de volta deve reproduzir o strip
        recovered = np.concatenate([p.image for p in output_pages], axis=0)
        self.assertEqual(recovered.shape, strip_img.shape)
        self.assertTrue(np.array_equal(recovered, strip_img))


class PasteBandsIntoStripTests(unittest.TestCase):
    def test_paste_bands_into_strip_modifies_band_y_range_only(self):
        from strip.reassemble import paste_bands_into_strip
        from strip.types import VerticalStrip, Band, Balloon, BBox
        import numpy as np

        strip = VerticalStrip(
            image=np.zeros((500, 300, 3), dtype=np.uint8),
            width=300, height=500, source_page_breaks=[0, 500],
        )
        rendered = np.full((100, 300, 3), 255, dtype=np.uint8)
        band = Band(
            y_top=200, y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(50, 220, 150, 280), confidence=0.9)],
            rendered_slice=rendered,
        )

        paste_bands_into_strip(strip, [band])

        # Linhas dentro da banda foram sobrescritas
        self.assertTrue(np.all(strip.image[250, :] == 255))
        # Linhas FORA da banda continuam zero
        self.assertTrue(np.all(strip.image[100, :] == 0))
        self.assertTrue(np.all(strip.image[400, :] == 0))

