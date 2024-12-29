"""
Common definitions for modules in the Ultimate RVC project that
facilitate training voice models.
"""


def get_gpu_info() -> list[tuple[str, int]]:
    """
    Retrieve information on locally available GPUs.

    Returns
    -------
    list[tuple[str, int]]
        A list of tuples containing the name and index of each locally
        available GPU.

    """
    # NOTE The lazy_import function does not work with torch
    # so we import it here manually
    import torch  # noqa: PLC0415

    ngpu = torch.cuda.device_count()
    gpu_infos: list[tuple[str, int]] = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024  # type: ignore[ReportUnknownMembershipType]
                + 0.4,
            )
            gpu_infos.append((f"{gpu_name} ({mem} GB)", i))
    return gpu_infos
