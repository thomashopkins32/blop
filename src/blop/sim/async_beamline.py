import numpy as np
import scipy as sp # type: ignore
from ophyd_async.core import StandardReadable, soft_signal_r_and_setter, StandardReadableFormat
from bluesky.protocols import Triggerable
from ..utils import get_beam_stats


class SimDetector(StandardReadable, Triggerable):
    def __init__(self, noise: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.add_children_as_readables():
            self.max, self.set_max = soft_signal_r_and_setter(float)
            self.area, self.set_area = soft_signal_r_and_setter(float)
            self.image, self.set_image = soft_signal_r_and_setter(np.ndarray)
            self.image_shape, self.set_image_shape = soft_signal_r_and_setter(np.ndarray, initial_value=np.array([300, 400]))
            self.noise, self.set_noise = soft_signal_r_and_setter(bool, initial_value=noise)

        with self.add_children_as_readables(format=StandardReadableFormat.HINTED_SIGNAL):
            self.sum, self.set_sum = soft_signal_r_and_setter(float)
            self.cen_x, self.set_cen_x = soft_signal_r_and_setter(int)
            self.cen_y, self.set_cen_y = soft_signal_r_and_setter(int)
            self.wid_x, self.set_wid_x = soft_signal_r_and_setter(int)
            self.wid_y, self.set_wid_y = soft_signal_r_and_setter(int)

    async def trigger(self):
        raw_image = await self.generate_beam(noise=await self.noise.get_value())

        current_frame = next(self._counter)

        image_shape = await self.image_shape.get_value()

        self._dataset.resize((current_frame + 1, *image_shape))
        self._dataset.get()[current_frame, :, :] = raw_image


        stats = get_beam_stats(raw_image)

        await self.set_max(stats["max"])
        await self.set_sum(stats["sum"])
        await self.set_cen_x(stats["cen_x"])
        await self.set_cen_y(stats["cen_y"])
        await self.set_wid_x(stats["wid_x"])
        await self.set_wid_y(stats["wid_y"])

        await self.set_image(raw_image)

        return None

    async def generate_beam(self, noise: bool = True):
        nx, ny = self.image_shape.get_value()

        x = np.linspace(-10, 10, ny)
        y = np.linspace(-10, 10, nx)
        X, Y = np.meshgrid(x, y)

        kbh_ush = await self.parent.kbh_ush.get_value()
        kbh_dsh = await self.parent.kbh_dsh.get_value()
        kbv_usv = await self.parent.kbv_usv.get_value()
        kbv_dsv = await self.parent.kbv_dsv.get_value()

        x0 = kbh_ush - kbh_dsh
        y0 = kbv_usv - kbv_dsv
        x_width = np.sqrt(0.2 + 5e-1 * (kbh_ush + kbh_dsh - 1) ** 2)
        y_width = np.sqrt(0.1 + 5e-1 * (kbv_usv + kbv_dsv - 2) ** 2)

        beam = np.exp(-0.5 * (((X - x0) / x_width) ** 4 + ((Y - y0) / y_width) ** 4)) / (
            np.sqrt(2 * np.pi) * x_width * y_width
        )

        ssa_inboard = await self.parent.ssa_inboard.get_value()
        ssa_outboard = await self.parent.ssa_outboard.get_value()
        ssa_lower = await self.parent.ssa_lower.get_value()
        ssa_upper = await self.parent.ssa_upper.get_value()

        mask = X > ssa_inboard
        mask &= X < ssa_outboard
        mask &= Y > ssa_lower
        mask &= Y < ssa_upper
        mask = sp.ndimage.gaussian_filter(mask.astype(float), sigma=1)

        image = beam * mask

        if noise:
            kx = np.fft.fftfreq(n=len(x), d=0.1)
            ky = np.fft.fftfreq(n=len(y), d=0.1)
            KX, KY = np.meshgrid(kx, ky)

            power_spectrum = 1 / (1e-2 + KX**2 + KY**2)

            white_noise = 1e-3 * np.random.standard_normal(size=X.shape)
            pink_noise = 1e-3 * np.real(np.fft.ifft2(power_spectrum * np.fft.fft2(np.random.standard_normal(size=X.shape))))
            # background = 5e-3 * (X - Y) / X.max()

            image += white_noise + pink_noise

        return image


class Beamline(StandardReadable):
    def __init__(self, noise: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.det = SimDetector(noise=noise)
        with self.add_children_as_readables(format=StandardReadableFormat.HINTED_SIGNAL):
            self.kbh_ush, self.set_kbh_ush = soft_signal_r_and_setter(float)
            self.kbh_dsh, self.set_kbh_dsh = soft_signal_r_and_setter(float)
            self.kbv_usv, self.set_kbv_usv = soft_signal_r_and_setter(float)
            self.kbv_dsv, self.set_kbv_dsv = soft_signal_r_and_setter(float)
            self.ssa_inboard, self.set_ssa_inboard = soft_signal_r_and_setter(float, initial_value=-5.0)
            self.ssa_outboard, self.set_ssa_outboard = soft_signal_r_and_setter(float, initial_value=5.0)
            self.ssa_lower, self.set_ssa_lower = soft_signal_r_and_setter(float, initial_value=-5.0)
            self.ssa_upper, self.set_ssa_upper = soft_signal_r_and_setter(float, initial_value=5.0)
