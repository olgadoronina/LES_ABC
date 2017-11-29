from params import *
from lxml import etree
# yhttp://lxml.de/tutorial.html
# http://www.diveintopython3.net/xml.html


class VTK_XML_Serial_Structured:
    def __init__(self, extent):
        extent_str = '{} {} {} {} {} {}'.format(extent[0], extent[1]-1, extent[2], extent[3]-1, extent[4], extent[5]-1)
        self.root = etree.Element('VTKFile', type='StructuredGrid', version='0.1', byte_order="LittleEndian")
        grid_type = etree.SubElement(self.root, 'StructuredGrid', WholeExtent=extent_str)
        piece = etree.SubElement(grid_type, 'Piece', Extent=extent_str)
        self.pointsdata = etree.SubElement(piece, 'PointData')
        self.celldata = etree.SubElement(piece, 'CellData')
        self.coord = etree.SubElement(piece, 'Points')

    def __str__(self):
        logging.info(etree.tostring(self.root, pretty_print=True, xml_declaration=True))


    def coords_to_string(self, X):
        string = str()
        a, Nx, Ny, Nz = X.shape
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    string += '{} {} {}\n'.format(repr(X[0, i, j, k]), repr(X[1, i, j, k]), repr(X[2, i, j, k]))
        return string

    def data_to_string(self, a):
        string = str()
        Nx, Ny, Nz = a.shape
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    string += '{}\n'.format(repr(a[i, j, k]))
        return string

    def add_pointdata(self, data, name):
        dataarray = etree.SubElement(self.pointsdata, 'DataArray', Scalar=name, type='Float32', format="ascii", NumberOfComponents="1")
        dataarray.text = self.data_to_string(data)

    def add_points(self, coord):
        coords = etree.SubElement(self.coord, 'DataArray', type='Float32', format="ascii", NumberOfComponents="3")
        coords.text = self.coords_to_string(coord)


    def write_to_file(self, filename, X, U):
        # self.add_points(X)
        # self.add_pointdata(U[0], 'u')
        # self.add_pointdata(U[1], 'v')
        # self.add_pointdata(U[2], 'w')
        doc = etree.ElementTree(self.root)
        doc.write(filename, pretty_print=True, xml_declaration=True)



